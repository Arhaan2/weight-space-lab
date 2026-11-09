# IFEval browser candidate recovery comparison

Arhaan2. The candidate is selected under the frozen strict prompt-compliance rule. This is exploratory on-device workflow validation; editor isolation remains unverified. All 128 held-out examples remain unexecuted.

The [original baseline](../ifeval-browser-v1/README.md) and [earlier stopped-stage report](../ifeval-browser-v1-candidate/README.md) are preserved. The first conversion-stage package check failed because the launcher dereferenced the virtual-environment interpreter symlink to system Python. A separately versioned helper now makes the configured interpreter absolute without dereferencing it and uses that same executable for environment verification and conversion. The user explicitly authorized one manual continuation; this was not an automatic retry. The imported checkpoint was reused without another import.

Offline verification through the production helper confirmed the configured virtual environment remained active after changing subprocess directory: the invoked executable and observed sys.executable matched, sys.prefix was the expected environment, and sys.prefix differed from sys.base_prefix. All 75 frozen dependency pins matched; 16 focused tests passed, including a real temporary virtual-environment regression test with an environment-only module. The actual conversion command used that verified executable. No installations or dependency upgrades were performed.
Frozen converter packages: `apache-tvm-ffi==0.1.13`, `mlc-ai-nightly-cpu==0.26.dev246`, `mlc-llm-nightly-cpu==0.26.dev6`.
Original launcher SHA-256: `d1554e04cfd16137295026b1bde7a899e79d6f78abd3e19eb39a8aacd6d1fe89`. Recovery launcher SHA-256: `b40f6318db0ab7c4acdac0f60b249f81e4c7af54d8056bcb6b874453107d2b69`. Base Git revision: `0f234cf2208544bbc0344e5086320b490bcc0ab5`. The repair is separately hash-bound; baseline frozen provenance was not rewritten.
An initial offline import probe omitted the existing production cache setting and encountered a cache-directory permission error. Its diagnostic was preserved; final environment verification reused the unchanged production cache/environment. This offline probe was not conversion or model inference.

| Compliance metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Strict prompt (primary) | 16/32 (50.00%) | 17/32 (53.12%) | +3.125 pp |
| Strict instruction | 16/32 (50.00%) | 17/32 (53.12%) | +3.125 pp |
| Loose prompt | 17/32 (53.12%) | 18/32 (56.25%) | +3.125 pp |
| Loose instruction | 17/32 (53.12%) | 18/32 (56.25%) | +3.125 pp |

Strict outcomes: 1 improved, 0 regressed, 31 tied, 0 unresolved. Loose outcomes: 1 improved, 0 regressed, 31 tied, 0 unresolved.
The descriptive paired bootstrap 95% interval for the strict difference is +0.000 to +9.375 percentage points (10,000 item resamples, seed 0). It covers this preselected development comparison; it does not establish a causal explanation or held-out generalization.

| Execution metric | Baseline | Candidate |
| --- | ---: | ---: |
| Natural completions | 20 | 20 |
| Length truncations | 12 | 12 |
| Unresolved grader errors | 0 | 0 |
| Scored output tokens | 13,941 | 13,903 |
| Scored input tokens | 6,123 | 6,123 |
| Scored generation seconds | 263.545 | 266.793 |
| Load seconds | 2.629 | 130.705 |
| Load through cleanup seconds | 273.364 | 405.731 |

The saved source audit found 122,880 targeted coordinates, 122,880 stored-bit changes and 122,880 numerical changes; 486 other tensors remained unchanged. These source changes are reported separately from converted quantization differences.

Candidate checkpoint SHA-256: `f9c767c06ec3c17dbdfedc43b695d64d025ae7393d3f425cee71ef45c358d6bf`.
Candidate deployed artifact SHA-256: `6e18cc4906ce18786b9197bcc7fb5f93ffc706d808b0809150349546eef3627b`.

| Source tensor | Targeted | Stored changes | Numerical changes | Decoded q4 changes | Outside target columns | Outside target groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `model.language_model.layers.15.mlp.down_proj.weight` | 61,440 | 61,440 | 61,440 | 51,863 | 3,043 | 0 |
| `model.language_model.layers.19.mlp.down_proj.weight` | 61,440 | 61,440 | 61,440 | 53,304 | 2,945 | 0 |

| Quantized tensor | Weight words changed | Nibbles changed | Scales changed | Decoded max absolute change | Decoded L2 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model.language_model.layers.15.mlp.down_proj.weight` | 2,028 | 5,557 | 1,914 | 0.0458984375 | 0.530728307 |
| `model.language_model.layers.19.mlp.down_proj.weight` | 2,057 | 5,622 | 1,894 | 0.044921875 | 0.45942111 |

The deployed parameter contract verified all 435 entries; 431 stayed byte-identical. Across both affected source tensors, 105,167 decoded values changed, including 5,988 outside exact source columns and 0 outside their quantization groups.

The source and deployed-artifact audits were verified before generation. Quantization can change values outside the exact source columns within their 32-coordinate groups; group spillover is disclosed above. Unchanged tokenizer, compiled model library and runtime configuration were retained. Full target columns and decoded tensor identities appear in [result.json](result.json).

| Resource observation (sampled maximum) | Baseline | Candidate |
| --- | ---: | ---: |
| Worker GPUBuffer descriptor bytes | 1,747,698,604 | 1,747,698,604 |
| Worker WASM address-space bytes | 283,181,056 | 283,181,056 |
| Worker JS heap bytes | unavailable | unavailable |
| Worker timer delay ms | 113.900 | 73,731 |

GPUBuffer totals measure descriptor accounting; WASM bytes measure address space. Neither is physical VRAM. Owned-process RSS/footprint maxima and host-wide ranges remain separately available in the JSON. Process measurements may include shared memory, host measurements include unrelated workloads, and sampled maxima do not establish absolute peaks. One shared worker series is counted once per run, not repeated per output. Timing and resource differences come from separate local runs with host activity; they do not isolate the weight edit as their cause.

Candidate capacity consumed: 1/1 engine-load attempt, 1/1 warmup, 32/32 scored dispatches, 33/33 total dispatches, and 13,915/32,784 reported generated tokens including warmup. Automatic generation retries: 0; pending requests: 0; limit violations: 0. The original successful import remains the sole consumed submission. The earlier failed precheck and this separately authorized manual conversion continuation are both preserved.
Four injected empty-thinking prefix IDs count toward max_tokens but not reported completion tokens. A 1024 cap permits at most 1020 sampled tokens; the 16-token warmup cap permits at most 12. Reservations retain 1024/16 bounds.
The converter retained its existing 1,800-second subprocess limit and 30-second package check. Conversion preparation took 29.880 seconds and packaging 8.742 seconds. Browser limits remained 120 seconds per request and 5,400 seconds from engine load through cleanup.
Candidate execution status: `completed`; runtime failure recorded: `False`. Cleanup: unload `True`, worker termination `True`, browser OS exit `True`, local harness exit `True`, remaining owned processes 0. Ordinary failed instructions and truncations remained results under the unchanged grading policy.

The frozen comparison retained all 32 development examples in the same order, the same pristine origin, tokenizer, chat template, quantization, browser runtime, warmup, answer extraction and official grading. Thinking was disabled; temperature 0, top-p 1, per-request seed 42, context 4,096 and output cap 1,024 remained unchanged. There were no baseline reruns, held-out generations, new proposals, or experimental frontier requests.

Frozen selection policy: Compare all 32 fixed development items; prefer candidate only for strictly higher strict prompt compliance with complete technical execution and grading; ties retain pristine. Technical execution and grading complete: `True`.

The supplied exploratory proposal used file-reading and JSON-parsing tools to ingest the development packet, with no external research, additional weight inspections, or target forward passes in that proposal session. Its selected columns are not a verified causal explanation for repetition. The original request and provenance remain separate, and editor isolation remains unverified.

The [independent recovery verification](verification.json) passed 44/44 checks. The original baseline and stopped-stage records remain historical evidence; this report is a new result.

Per-item results use 1-based frozen schedule positions. Pass/fail refer to official compliance, not technical execution success.

| Position | Strict baseline → candidate | Strict change | Loose baseline → candidate | Loose change | Completion baseline → candidate |
| ---: | --- | --- | --- | --- | --- |
| 1 | pass → pass | tied | pass → pass | tied | ok → ok |
| 2 | pass → pass | tied | pass → pass | tied | ok → ok |
| 3 | fail → fail | tied | fail → fail | tied | ok → ok |
| 4 | pass → pass | tied | pass → pass | tied | truncated → truncated |
| 5 | fail → fail | tied | fail → fail | tied | truncated → truncated |
| 6 | pass → pass | tied | pass → pass | tied | ok → ok |
| 7 | fail → fail | tied | fail → fail | tied | truncated → ok |
| 8 | fail → fail | tied | fail → fail | tied | truncated → truncated |
| 9 | pass → pass | tied | pass → pass | tied | ok → ok |
| 10 | fail → fail | tied | fail → fail | tied | ok → ok |
| 11 | pass → pass | tied | pass → pass | tied | truncated → ok |
| 12 | pass → pass | tied | pass → pass | tied | ok → ok |
| 13 | pass → pass | tied | pass → pass | tied | ok → ok |
| 14 | fail → fail | tied | fail → fail | tied | truncated → truncated |
| 15 | fail → fail | tied | fail → fail | tied | ok → ok |
| 16 | pass → pass | tied | pass → pass | tied | ok → ok |
| 17 | fail → pass | improved | fail → pass | improved | truncated → truncated |
| 18 | pass → pass | tied | pass → pass | tied | truncated → truncated |
| 19 | fail → fail | tied | fail → fail | tied | ok → ok |
| 20 | pass → pass | tied | pass → pass | tied | ok → ok |
| 21 | pass → pass | tied | pass → pass | tied | ok → ok |
| 22 | pass → pass | tied | pass → pass | tied | ok → truncated |
| 23 | pass → pass | tied | pass → pass | tied | ok → ok |
| 24 | fail → fail | tied | fail → fail | tied | ok → ok |
| 25 | fail → fail | tied | pass → pass | tied | truncated → truncated |
| 26 | fail → fail | tied | fail → fail | tied | ok → ok |
| 27 | fail → fail | tied | fail → fail | tied | truncated → truncated |
| 28 | fail → fail | tied | fail → fail | tied | ok → ok |
| 29 | pass → pass | tied | pass → pass | tied | ok → truncated |
| 30 | pass → pass | tied | pass → pass | tied | ok → ok |
| 31 | fail → fail | tied | fail → fail | tied | truncated → truncated |
| 32 | fail → fail | tied | fail → fail | tied | truncated → truncated |

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2). Existing BrowserLLM work, Gufran's contribution, upstream licenses and scientific attribution remain preserved. Sources: [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), [pinned Google Research IFEval](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval). Private prompts, packets, response, authorization and machine paths are excluded.
