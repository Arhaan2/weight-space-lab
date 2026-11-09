# IFEval browser candidate stage — stopped before conversion

Arhaan2. The proposal import and saved-checkpoint audit succeeded. The conversion stage failed during its package metadata check, before the converter subprocess began. No candidate engine load or generation occurred, so the matched development comparison is unavailable and no arm was selected. All 128 held-out examples remain unexecuted.

The conversion wrapper resolves the configured virtual-environment Python symlink to the base/system interpreter. That interpreter lacks `apache-tvm-ffi` package metadata. Read-only diagnosis confirmed that the configured virtual environment itself contains the exact frozen package versions. The command file exited with code 1. Execution stopped under the explicit no-retry authorization; no frozen implementation was changed.

The one import slot was consumed successfully in 17.307 seconds of its 900-second limit. There were no import failures, pending requests, or retries. The source audit passed, with 486 tensors unchanged.

Pristine checkpoint SHA-256: `f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d`.
Candidate checkpoint SHA-256: `f9c767c06ec3c17dbdfedc43b695d64d025ae7393d3f425cee71ef45c358d6bf`.
Candidate deployed-artifact hash: **unavailable**. No converted or packaged artifact was produced; quantized and decoded quantized differences were not measured.

| Source tensor | Targeted coordinates | Stored-bit changes | Numerical changes |
| --- | ---: | ---: | ---: |
| `model.language_model.layers.15.mlp.down_proj.weight` | 61,440 | 61,440 | 61,440 |
| `model.language_model.layers.19.mlp.down_proj.weight` | 61,440 | 61,440 | 61,440 |
| Total | 122,880 | 122,880 | 122,880 |

Target support: `model.language_model.layers.15.mlp.down_proj.weight` columns 0–59 scaled by 0.75; `model.language_model.layers.19.mlp.down_proj.weight` columns 0–59 scaled by 0.75.

Actual stored changes comprise 0.01406853% of all 873,438,784 stored coordinates. Bit-only changes: 0. Maximum absolute numerical change: 0.0458984375; L2 change: 0.678826476073. These are measured saved BF16 changes, separate from unavailable quantization results. Every targeted coordinate happened to change in this saved checkpoint; that does not establish what would survive conversion.

The [original baseline report](../ifeval-browser-v1/README.md) remains preserved and was not rerun. Candidate scores are unavailable, not zero scores.

| Compliance metric | Completed baseline | Candidate |
| --- | ---: | --- |
| Strict prompt (primary) | 16/32 (50.00%) | Unavailable |
| Strict instruction | 16/32 (50.00%) | Unavailable |
| Loose prompt | 17/32 (53.12%) | Unavailable |
| Loose instruction | 17/32 (53.12%) | Unavailable |

Per-item improvements, regressions and ties are unmeasured for all 32 pairs. No paired effect, confidence interval, or compliance improvement can be reported.

The existing baseline had 20 natural completions and 12 truncations, with 0 unresolved grader errors. It used 13,941 scored output tokens over 263.545 seconds. Candidate natural completions, truncations and scored output tokens are all zero because no requests ran; candidate generation time and browser elapsed time were not measured.

Actual candidate capacity consumed: one successful import, one conversion-stage attempt that failed in its package precheck, zero conversion subprocesses, zero packaging operations, zero engine-load attempts, zero warmups, zero scored requests, zero total generation dispatches, and zero generated tokens. There were zero automatic retries, new weight proposals, baseline reruns, held-out generations, or experimental frontier requests. The fresh browser generation ledger was never created or spent because authorization binding was not reached.

The existing conversion subprocess limit remained 1,800 seconds, with a 30-second package precheck; it was not replaced by the browser limit. The authorized browser bounds remained one load, one warmup of at most 16 tokens, 32 scored requests of at most 1,024 tokens, 33 total dispatches, 32,784 total tokens, 120 seconds per request, and 5,400 seconds from load through cleanup. The browser deadline never started.

The frozen profile still requires the same 32 development examples and order, pristine origin, tokenizer, chat template, q4f16_1 quantization, WebLLM runtime, warmup policy, extraction and official grader; thinking disabled; temperature 0; top-p 1; seed 42 reset per request; context 4,096; output cap 1,024. No prompt, decoding, output-cleanup, grading or task-selection adjustment was made.

Candidate resource observations are unavailable because no browser run began. Existing baseline worker GPUBuffer descriptor accounting peaked at 1,747,698,604 bytes and WASM address space at 283,181,056 bytes; JS heap was unavailable. These are not physical VRAM measurements. The JSON retains baseline host-wide ranges and maximum single observed process per role separately: host readings include unrelated workloads, RSS/footprint can include shared memory, and sampled maxima do not establish absolute peaks.

Cleanup verification found the command had exited and no process matched the known checkpoint, converter, or experiment-runner command signatures. The import worker completed. Browser unload and browser cleanup are not applicable because no candidate browser started. This check does not inventory unrelated user browsers.

Frozen selection policy: Compare all 32 fixed development items; prefer candidate only for strictly higher strict prompt compliance with complete technical execution and grading; ties retain pristine. Status: **not evaluable**; selected arm: **none**, because candidate deployment, execution and grading did not complete. The existing baseline remains available, with no measured comparison decision.

The supplied proposal provenance records file-reading and JSON-parsing tools used to ingest the development packet, with no external research, additional weight inspections, or target forward passes in that proposal session. The session was not tool-free. The selected columns are not a verified causal explanation for repetition. The original request is preserved separately from this provenance statement. Editor isolation remains unverified.

The [final independent verifier](verification-final.json) passed all 22 checks, including 578 current evidence pins, 1,387 available historical pins, and independent recounting of actual BF16 changes. The 300 previously unavailable historical files remain outside the preservation claim. The [initial verification](verification.json) is preserved: its sole checker failure expected nested package-error text in the outer pipeline log. The corrected checker verifies the recorded diagnostic location. The final verifier additionally removes redundant private request-envelope identifiers from its source and derives exact checks from preserved evidence. Both prior verification outputs remain preserved; no experiment was rerun.

Per-item results use the 1-based frozen schedule position, keeping source identifiers and private prompts out of this public report.

| Position | Baseline strict | Baseline loose | Baseline completion | Candidate | Paired change |
| ---: | --- | --- | --- | --- | --- |
| 1 | pass | pass | ok | Unexecuted | Unmeasured |
| 2 | pass | pass | ok | Unexecuted | Unmeasured |
| 3 | fail | fail | ok | Unexecuted | Unmeasured |
| 4 | pass | pass | truncated | Unexecuted | Unmeasured |
| 5 | fail | fail | truncated | Unexecuted | Unmeasured |
| 6 | pass | pass | ok | Unexecuted | Unmeasured |
| 7 | fail | fail | truncated | Unexecuted | Unmeasured |
| 8 | fail | fail | truncated | Unexecuted | Unmeasured |
| 9 | pass | pass | ok | Unexecuted | Unmeasured |
| 10 | fail | fail | ok | Unexecuted | Unmeasured |
| 11 | pass | pass | truncated | Unexecuted | Unmeasured |
| 12 | pass | pass | ok | Unexecuted | Unmeasured |
| 13 | pass | pass | ok | Unexecuted | Unmeasured |
| 14 | fail | fail | truncated | Unexecuted | Unmeasured |
| 15 | fail | fail | ok | Unexecuted | Unmeasured |
| 16 | pass | pass | ok | Unexecuted | Unmeasured |
| 17 | fail | fail | truncated | Unexecuted | Unmeasured |
| 18 | pass | pass | truncated | Unexecuted | Unmeasured |
| 19 | fail | fail | ok | Unexecuted | Unmeasured |
| 20 | pass | pass | ok | Unexecuted | Unmeasured |
| 21 | pass | pass | ok | Unexecuted | Unmeasured |
| 22 | pass | pass | ok | Unexecuted | Unmeasured |
| 23 | pass | pass | ok | Unexecuted | Unmeasured |
| 24 | fail | fail | ok | Unexecuted | Unmeasured |
| 25 | fail | pass | truncated | Unexecuted | Unmeasured |
| 26 | fail | fail | ok | Unexecuted | Unmeasured |
| 27 | fail | fail | truncated | Unexecuted | Unmeasured |
| 28 | fail | fail | ok | Unexecuted | Unmeasured |
| 29 | pass | pass | ok | Unexecuted | Unmeasured |
| 30 | pass | pass | ok | Unexecuted | Unmeasured |
| 31 | fail | fail | truncated | Unexecuted | Unmeasured |
| 32 | fail | fail | truncated | Unexecuted | Unmeasured |

See [detailed measurements and audit identities](result.json). Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2). Existing BrowserLLM work, Gufran's contribution, upstream licenses and scientific attribution remain preserved. Sources: [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), [pinned Google Research IFEval](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval). Private prompts, packets, response, authorization and machine paths are excluded from this report.
