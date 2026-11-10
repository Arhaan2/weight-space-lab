# IFEval browser development baseline

Arhaan2. The pristine Qwen3.5-0.8B baseline completed locally using WebLLM 0.2.86, WebGPU, and the existing q4f16_1 artifact. This is an exploratory development measurement; editor isolation remains unverified.

| Metric | Passed / scheduled | Compliance |
| --- | ---: | ---: |
| Strict prompt (primary) | 16 / 32 | 50.00% |
| Strict instruction | 16 / 32 | 50.00% |
| Loose prompt | 17 / 32 | 53.12% |
| Loose instruction | 17 / 32 | 53.12% |

All 32 scheduled development prompts remain in the denominator: 20 natural completions, 12 length truncations, 0 unexecuted, and 0 unresolved grader errors. Truncated text was graded under the frozen policy. No response retries or cleanup rewrites occurred.

Strict failures by instruction type: `length_constraints:nth_paragraph_first_word` 2, `combination:two_responses` 2, `detectable_format:number_bullet_lists` 2, `change_case:english_lowercase` 2, `change_case:capital_word_frequency` 1, `detectable_content:number_placeholders` 1, `change_case:english_capital` 1, `detectable_format:number_highlighted_sections` 1, `keywords:existence` 1, `startend:quotation` 1, `startend:end_checker` 1, `punctuation:no_comma` 1.

Scored generation used 13,941 reported completion tokens and 6,123 input tokens over 263.545 seconds. Engine loading took 2.629 seconds. First durable load grant through final cleanup took 273.364 seconds of 5,400.

Consumption: one engine-load attempt, one warmup, 32 scored dispatches, 13,953 of 32,784 allowed reported completion tokens, and no retries. The warmup was capped at 16 tokens and accounted separately. WebLLM's four injected empty-thinking prefix IDs count toward `max_tokens` but not reported sampled tokens; the 1,024-token setting therefore permits at most 1,020 sampled tokens. Candidate, held-out, and experimental frontier consumption remain zero.

The newly frozen data contains 32 Ai2 practice prompts (32 instructions) and a separate 128-prompt Google IFEval subset (173 instructions). Both cover 18 instruction types across eight families. Pre-output screening excluded 4,563 practice records and 197 Google records. Selection uses seed 42 and balances supported instruction types with stable hashed tie-breaking. Exact/normalized selected prompts are disjoint; semantic paraphrases and model pretraining exposure remain unverified. Prior Google study partitions were preserved.

The actual compiled WASM supports the effective 4,096-token context, verified from embedded metadata and runtime source. Exact deployed-tokenizer counts agree for all 160 selected prompts. Development input spans 44–1,644 tokens and evaluation input 42–238 tokens; every input plus the 1,024-token output allowance fits. This feasibility check was frozen before any new output.

Observed worker descriptor accounting peaked at 1,747,698,604 GPUBuffer bytes and 283,181,056 WASM address-space bytes. Neither is physical VRAM. JS heap was unavailable. Host-wide readings and owned-process footprints are reported separately in the result, with shared-memory and sampling limits. Engines unloaded, the worker and dedicated browser exited, and no owned process remained. The collector's Xcode loader warning did not prevent scoped resource sampling.

The private Astra/Ultra packet contains all development responses, independent per-constraint grades, recorded bounded tensor statistics, and exact subsequent import/conversion/comparison commands. Its resource appendix is joined to immutable execution evidence. Requested editor identity is separate from unavailable observed model/version/effort/usage/billing. The original proposal is still required; no placeholder or integration edit was supplied.

The existing official Google checkers, independent grading and paired-analysis pipeline, checkpoint editor, conversion audits, browser v2 asset serving, and resource watchdog are reused. The Ai2 adapter has explicit supported mappings and 42 passing official known-answer checks. A separate read-only verifier passed all 42 evidence checks, including the dispatch ledger, raw/grade bindings, packet resources, and 1,387 unchanged prior evidence files. The 300 historically unavailable files remain explicitly outside that byte-verification claim. See [independent verification](verification.json), [detailed results](result.json), and the [manual handoff procedure](../../docs/ifeval-manual-handoff.md).

Sources: [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), [pinned Google IFEval](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval). Upstream licenses, scientific authorship, and Gufran's contribution remain preserved. Private prompts, packets, machine paths and session records are excluded from public artifacts.
