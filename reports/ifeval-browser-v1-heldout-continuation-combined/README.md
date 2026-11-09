# 128-example IFEval on-device subset — continued comparison

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

The candidate has higher observed strict prompt compliance on this subset. The paired 95% interval includes zero, so the direction remains uncertain.

Status: **complete_manually_continued**. The frozen candidate comparison joins 128 preserved baseline grades, 94 preserved candidate-prefix grades, and only the newly graded 34-position candidate suffix. The candidate spans two sessions with an extra warm-up; this does not constitute successful completion of the original uninterrupted protocol.

| Compliance | Baseline | Candidate | Difference (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: | ---: | --- |
| Strict prompt (primary) | 69/128 (53.91%) | 70/128 (54.69%) | +0.78 | [-5.47, +7.03] |
| Loose prompt | 74/128 (57.81%) | 77/128 (60.16%) | +2.34 | [-3.91, +8.59] |
| Strict instruction | 106/173 (61.27%) | 108/173 (62.43%) | +1.16 | [-3.85, +6.29] |
| Loose instruction | 112/173 (64.74%) | 116/173 (67.05%) | +2.31 | [-2.91, +7.69] |

Counts are aggregated across the disjoint prefix and suffix; their percentages are never averaged. All 128 prompts and 173 instructions per arm remain in the denominators. Intervals use the frozen 10,000 paired percentile bootstrap resamples, seed 0, with whole prompts as clusters. Instruction ratios are recomputed from compliant and scheduled instruction counts in each resample. No exact paired significance test was added, and significance is not a completion condition.

| Paired outcomes | Improvements | Regressions | Ties | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Strict prompt | 8 | 7 | 113 | 0 |
| Loose prompt | 10 | 7 | 111 | 0 |
| Strict instruction | 11 | 9 | 153 | 0 |
| Loose instruction | 12 | 8 | 153 | 0 |

Instruction pairs refer to corresponding constraints within each frozen prompt. Their uncertainty continues to use prompt clusters; instructions are not treated as independent bootstrap units.

| Arm observation | Baseline (128 prompts) | Candidate combined (128 prompts) |
| --- | ---: | ---: |
| Natural completions | 69 | 77 |
| Truncations | 59 | 51 |
| Unexecuted slots | 0 | 0 |
| Dispatched without output | 0 | 0 |
| Ambiguous dispatch outcomes | 0 | 0 |
| Attempted model failures | 0 | 0 |
| Grader errors | 0 | 0 |
| Grader timeouts | 0 | 0 |
| Scored sampled tokens | 68,243 | 60,189 |
| Scored generation seconds | 1,704.563 | 1,227.986 |

| Session usage | Baseline | Candidate interrupted | Candidate continuation |
| --- | ---: | ---: | ---: |
| Loads | 1 | 1 | 1 |
| Warm-ups | 1 | 1 | 1 |
| Scored dispatches | 128 | 94 | 34 |
| All generation dispatches | 129 | 95 | 35 |
| Known tokens including warm-up | 68,255 | 45,600 | 14,613 |
| Generation seconds including warm-up | 1,705.319 | 969.693 | 259.911 |
| Load through cleanup seconds | 1,740.213 | 1,022.176 | 269.835 |
| Natural scored completions | 69 | 55 | 22 |
| Scored truncations | 59 | 39 | 12 |
| Scored sampled tokens | 68,243 | 45,588 | 14,601 |
| Scored generation seconds | 1,704.563 | 968.893 | 259.093 |

Cumulative consumption: **3 loads, 3 warm-ups, 256 scored requests, 259 total dispatches**, and 128,468 known generated tokens including warm-ups. Token counts are unknown for 0 requests. Reservations: 262,192 tokens; conservative charged usage: 128,468. Automatic retries: 0; pending requests: 0; limit violations: 0.

Summed active load-through-cleanup time: 3,032.224 seconds. Candidate interruption gap: 75,009.339 seconds. First baseline load through final cleanup calendar span: 78,065.316 seconds. The original aggregate elapsed record remains 2,786.134 seconds. The continuation has its own load-through-cleanup clock capped at 5,400 seconds; the closed original 10,800-second clock and ledgers were not reopened or replenished.

The continuation allowance was one load, one warm-up capped at 16 tokens, 34 scored requests capped at 1,024 tokens each, at most 35 dispatches and 34,832 generated tokens, and 120 seconds per request. The expected completed-study total is three loads, three warm-ups and 259 dispatches. Warm-ups are excluded from compliance scores. Four injected empty-thinking prefix IDs count toward each request cap but not sampled completion usage, allowing at most 1,020 sampled scored tokens and 12 sampled warm-up tokens.

The original candidate session stopped after 94 scored completions when its supervisor used an event snapshot missing the durable predecessor completion and rejected the gate for position 95. The dispatch audit found no acknowledgement, reservation or model call for positions 95–128. The original session required forced termination; process closure was verified, but graceful unload was not confirmed. Its failure, grades, artifacts, closed ledger and timing records remain unchanged. The separately versioned continuation runner uses the prepared supervisor repair.

| Session cleanup | Status | Graceful unload | Worker / browser / harness closed | Remaining processes |
| --- | --- | --- | --- | ---: |
| Baseline | completed | True | True | 0 |
| Candidate interrupted | stopped | False | True | 0 |
| Candidate continuation | completed | True | True | 0 |

Recorded session-level execution failures: 1. The original supervisor failure remains part of the result even when all new suffix outputs complete. Per-session failure categories, cleanup confirmations, token accounting and resource observations are preserved in the JSON.

| Resource observation (sampled maximum) | Baseline | Candidate interrupted | Candidate continuation |
| --- | ---: | ---: | ---: |
| Worker GPUBuffer descriptor bytes | 1,747,697,720 | 1,747,697,720 | 1,747,697,720 |
| Worker WASM address-space bytes | 283,181,056 | 283,181,056 | 283,181,056 |
| Worker JS heap bytes | unavailable | unavailable | unavailable |
| Worker timer delay ms | 15,305.600 | 9,645.000 | 62.500 |

A developer-tool diagnostic was preserved privately. Host telemetry remained available in the continuation session; the diagnostic is not counted as an execution failure. Execution status and failures follow the terminal supervisor evidence.

A single sequential comparison with an interrupted and resumed candidate does not demonstrate a controlled speed or energy improvement. GPUBuffer descriptor allocations and WASM address space are not physical VRAM. CPU, disk and kernel-energy observations are cumulative counters since process creation and are reported as maximum single observed process values per role, not task-only deltas or sums. Kernel energy is not battery energy. Host activity, shared allocations and sample spacing limit these observations. All three sessions retain their own worker and host sample counts and numeric ranges in the JSON.

Frozen generation settings remain fresh conversations, thinking disabled, temperature 0, top-p 1, seed 42 reset per request, context 4,096, output cap 1,024, WebLLM 0.2.86 and q4f16_1. Ordinary incorrect and truncated responses are retained under the official frozen scoring policy.

The sole development improvement (schedule item 17) required at least 22 highlighted sections. The baseline echoed long plot text with 3 highlighted spans; the candidate repeatedly emitted the same short title, producing 169 complete highlighted spans before truncation. Both used 1,020 sampled tokens. The official strict and loose fail-to-pass grades remain unchanged. This gain reflects repetitive content satisfying the formatting checker; it does not demonstrate better substantive answering or reduced repetition.

The [original report](../ifeval-browser-v1-heldout/README.md) retains the interrupted result and its qualitative inspection of changed prefix outcomes. Official compliance grades do not establish substantive answer correctness.

All 34 newly completed candidate suffix outputs were compared with their preserved baseline grades. Two pairs changed at least one strict or loose instruction outcome, at schedule positions 98 and 114. Both are regressions under both prompt metrics; the remaining 32 pairs are ties, with no suffix improvements or additional instruction-only changes. These observations use the saved official grades without regrading or introducing a new scoring rule. Repetitive output is visible in both changed pairs. Compliance grades do not establish substantive answer correctness, and the English-and-capitals checker combines capitalization with automatic language identification.

- Schedule item 98: strict prompt regressed; loose prompt regressed. The baseline stops naturally after two short, similarly worded responses with the requested separator, receiving both official instruction credits. The candidate repeats quotation marks until truncation and never supplies the checked two-response structure. It retains forbidden-word credit. The regression reflects a loss of response structure within repetitive output; the baseline's checker pass does not establish that its two responses are substantively different or informative.
- Schedule item 114: strict prompt regressed; loose prompt regressed. Both responses repeat uppercase text until truncation. The baseline repeatedly cycles through a phrase, while the candidate collapses into repetition of a single word. The saved official English-and-capitals grade changes from pass to fail even though the candidate's visible text uses capital letters. This checker also performs automatic language identification; its internal language label was not saved, so the regression should not be described as an observed lowercase violation. Neither repetitive response demonstrates a substantive song with explanations, and the official grades remain unchanged.

Frozen identities:

- Authorized continuation plan: `f366b08d4d9a25d54307af6638b1c5fa66f8ee9b87174e087de18d1ad8f9dad8`.
- Original evaluation freeze: `89ed9703869ab7c8b1032b3d8c5fa59fefed0215d9d0341016684d60b39b8a27`.
- Continuation manifest: `0faa9ec2b548c5070106701fbe965640b44a678c1ee6b185c3d25aa7fcd4ab83`.
- Baseline source checkpoint: `f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d`.
- Baseline deployed artifact: `5ead3e9c60107ab7168bee5373f6cd636c839b82557fed6550aaaa30170aa5b0`.
- Candidate source checkpoint: `f9c767c06ec3c17dbdfedc43b695d64d025ae7393d3f425cee71ef45c358d6bf`.
- Candidate deployed artifact: `6e18cc4906ce18786b9197bcc7fb5f93ffc706d808b0809150349546eef3627b`.
- Frozen original runner: `cfbd51cbb79349442770022bb0dfcbb3a7b089debc3437bab9baeab69cca8bcf`.
- Continuation runner: `bda28582f9efbc04b2fd1cb624ce33baa5edbbc5db85097cf965caf79fb54e65`.
- Continuation supervisor repair: `5609efba2f7bf9ef334c75b8abdb2345e315dc32de8d3e5066d6cb25529a6c66`.
- Frozen continuation merger: `9ac7875ee4d76c75fe8bc48edbb99d6d1665b63da0e5240c416ac631638955f7`.

The [machine-readable report](result.json) retains all four endpoints, frozen uncertainty settings, session and cumulative accounting, and sanitized identities. The [independent execution verification](verification.json) audits the continuation and preserved evidence. Private merge references remain with the private evidence. Public artifacts contain no private prompts, outputs, approval text, editor packets or machine paths.

This remains an exploratory manual-editor study with editor isolation unverified. It evaluates 128 prompts, not the full 541-example IFEval benchmark, and does not establish representative full-benchmark performance or absence of pretraining exposure. The bootstrap describes one preselected comparison and does not include candidate-selection uncertainty. Instruction compliance is distinct from substantive answer correctness. The selected weight columns are not a verified causal explanation of response behavior. The subset is exposed and cannot be called untouched for future edits. No new proposal, checkpoint conversion, weight edit, baseline rerun, previously dispatched scored rerun, or development experiment was performed.

Proposal provenance (reported, not independently observed): Original exploratory proposal based on the supplied development packet. The proposal session used file-reading and JSON-parsing tools to ingest that packet. It used no external research, additional weight inspections, or target forward passes.

Upstream attribution remains with [Google Research IFEval (Zhou et al.)](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval), [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), Qwen, WebLLM/MLC, existing BrowserLLM work, Gufran’s contribution, and their preserved licenses.
