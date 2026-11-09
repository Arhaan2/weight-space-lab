# IFEval browser V1 — completed, inconclusive

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

**V1 is closed as a completed experiment with an inconclusive held-out compliance result.** Strict prompt compliance was **70/128 candidate versus 69/128 baseline: +0.78125 percentage points**, with the frozen **95% interval −5.47 to +7.03 percentage points**. **Improvement and equivalence are unestablished.** Completion includes the separately run candidate continuation; the original uninterrupted protocol failed.

This is the canonical closeout of the [completed combined report](../ifeval-browser-v1-heldout-continuation-combined/README.md). It preserves that report and its uncertainty analysis. It is a 128-prompt subset experiment, not the full 541-prompt benchmark. V2 is [proposed only](../../docs/ifeval-browser-v2-protocol.md).

## Official held-out results

| Official metric | Baseline | Candidate | Effect, candidate − baseline (pp) | Frozen 95% interval (pp) | Improvements / regressions / ties |
| --- | ---: | ---: | ---: | --- | ---: |
| Strict prompt (primary) | 69/128 (53.90625%) | 70/128 (54.68750%) | +0.78125 | [−5.47, +7.03] | 8 / 7 / 113 |
| Loose prompt | 74/128 (57.81250%) | 77/128 (60.15625%) | +2.34375 | [−3.91, +8.59] | 10 / 7 / 111 |
| Strict instruction | 106/173 (61.27168%) | 108/173 (62.42775%) | +1.15607 | [−3.85, +6.29] | 11 / 9 / 153 |
| Loose instruction | 112/173 (64.73988%) | 116/173 (67.05202%) | +2.31214 | [−2.91, +7.69] | 12 / 8 / 153 |

Every arm retains **128 prompts and 173 instructions**; unresolved pairs are zero. Paired counts come from corresponding saved outcomes, not from the aggregate net difference. Instruction ties include both-pass and both-fail constraints. Instruction compliance is the total compliant count divided by the total scheduled count, not an average of per-prompt percentages.

Uncertainty is the frozen 10,000-resample paired percentile bootstrap, seed 0, lexicographic item-ID order, whole prompts as clusters, and NumPy's default linear quantiles. Instruction ratios are recomputed within each prompt-cluster resample. No grader, exact paired significance test, equivalence margin, primary metric, or resampling method was added at closeout. Intervals describe one development-selected comparison and exclude selection uncertainty. Ordinary incorrect and truncated responses retain their saved official grades. Statistical significance was never a completion condition.

## Per-arm quality and resources

| Held-out observation | Baseline | Candidate, both sessions |
| --- | ---: | ---: |
| Scored prompts / instructions | 128 / 173 | 128 / 173 |
| Natural completions / truncations | 69 / 59 | 77 / 51 |
| Scored sampled output tokens | 68,243 | 60,189 |
| Scored input tokens | 8,745 | 8,745 |
| Scored generation seconds | 1,704.563 | 1,227.986 |
| Loads / warm-ups / total dispatches | 1 / 1 / 129 | 2 / 2 / 130 |
| Sampled tokens including warm-ups | 68,255 | 60,213 |
| Generation seconds including warm-ups | 1,705.319 | 1,229.604 |
| Active load-through-cleanup seconds | 1,740.213 | 1,292.011 |
| Session execution failures | 0 | 1 |
| Forced-termination sessions | 0 | 1 |
| Unexecuted / ambiguous / output-missing requests | 0 / 0 / 0 | 0 / 0 / 0 |
| Attempted model failures / grader errors / grader timeouts | 0 / 0 / 0 | 0 / 0 / 0 |

Candidate scored consumption is 8,054 fewer sampled tokens and 476.577 fewer generation seconds on these outputs. These are observations from sequential sessions with different output behavior and an interrupted/resumed candidate. They do not isolate a speed, energy, or useful-efficiency benefit. Aggregate study consumption is an experiment cost, not an arm efficiency estimator.

| Held-out session | Baseline | Candidate interrupted | Candidate continuation |
| --- | ---: | ---: | ---: |
| Scored dispatches | 128 | 94 | 34 |
| Loads / warm-ups / all dispatches | 1 / 1 / 129 | 1 / 1 / 95 | 1 / 1 / 35 |
| Natural / truncated scored completions | 69 / 59 | 55 / 39 | 22 / 12 |
| Scored sampled tokens | 68,243 | 45,588 | 14,601 |
| Sampled tokens including warm-up | 68,255 | 45,600 | 14,613 |
| Scored generation seconds | 1,704.563 | 968.893 | 259.093 |
| Generation seconds including warm-up | 1,705.319 | 969.693 | 259.911 |
| Load-through-cleanup seconds | 1,740.213 | 1,022.176 | 269.835 |
| Reserved tokens | 131,088 | 96,272 | 34,832 |
| Graceful unload confirmed | yes | no | yes |
| Owned process closure verified / remaining | yes / 0 | yes / 0 | yes / 0 |
| Worker / host samples | 1,684 / 336 | 996 / 202 | 282 / 53 |
| GPUBuffer descriptor bytes, sampled maximum | 1,747,697,720 | 1,747,697,720 | 1,747,697,720 |
| WASM address-space bytes, sampled maximum | 283,181,056 | 283,181,056 | 283,181,056 |
| Worker JS heap bytes | unavailable | unavailable | unavailable |
| Worker timer delay ms, sampled maximum | 15,305.600 | 9,645.000 | 62.500 |

GPUBuffer sizes are allocation-descriptor accounting, including pools but excluding textures, driver overhead and physical residency. WASM bytes measure address space. Neither measures physical VRAM. The saved [session resource records](../ifeval-browser-v1-heldout-continuation-combined/result.json) retain all host numeric ranges and per-role process maxima. RSS/footprint can include shared memory; host samples include unrelated work. CPU time, disk bytes and kernel-energy values are cumulative since process creation, summarized as the maximum single observed process per role, not task deltas or process sums. Kernel energy is not battery energy. Sampled maxima need not be true peaks; timer delay is a worker responsiveness signal, not token latency. Candidate process counters must remain separate across its sessions.

**Held-out cumulative accounting:** 3 loads, 3 warm-ups, 256 scored requests, 259 total dispatches; 128,432 scored sampled tokens, 36 warm-up tokens, **128,468 known/charged tokens**, 262,192 reserved tokens; 2,932.549 scored generation seconds, 2,934.923 including warm-ups, and 3,032.224 active load-through-cleanup seconds. Unknown-token requests, automatic retries, pending requests and limit violations are all zero. The 262,192 reservations are actual reservations across sessions, not the sum of authorized ceilings: the original 262,176 ceiling plus the new 34,832 continuation allowance was 297,008, with the original unused capacity closed.

The candidate interruption gap was 75,009.339 seconds. First baseline load to final cleanup spanned 78,065.316 calendar seconds. The original 2,786.134-second aggregate elapsed record and 10,800-second limit stayed closed; continuation had its own 5,400-second clock. Active session time, original aggregate elapsed, and calendar span are distinct quantities.

## Development and full V1 accounting

The frozen rule selected a technically complete candidate only if its strict prompt score exceeded pristine over the fixed 32-prompt development set; ties retained pristine. Strict prompt and strict instruction scores each moved **16/32 → 17/32**; loose prompt and loose instruction each moved **17/32 → 18/32**. There was **1 improvement, 0 regressions, 31 ties** in strict and loose prompt outcomes. The saved development strict interval was [0, +9.375] pp (10,000 paired resamples, seed 0), descriptive and not held-out evidence. The sole improvement used repetitive highlights to meet the checker.

| Development session | Baseline | Candidate |
| --- | ---: | ---: |
| Natural / truncated completions | 20 / 12 | 20 / 12 |
| Scored output / input tokens | 13,941 / 6,123 | 13,903 / 6,123 |
| Scored generation seconds | 263.545 | 266.793 |
| Load seconds | 2.629 | 130.705 |
| Load-through-cleanup seconds | 273.364 | 405.731 |
| Tokens including warm-up | 13,953 | 13,915 |
| Loads / warm-ups / scored / total dispatches | 1 / 1 / 32 / 33 | 1 / 1 / 32 / 33 |

**Full V1 target-browser consumption** (development plus completed held-out, excluding earlier unrelated experiments): **5 loads, 5 warm-ups, 320 scored requests, 325 dispatches, 156,276 scored sampled tokens and 156,336 including warm-ups**. Actual token reservations total 327,760. Summed active browser time is 3,711.320 seconds; generation time including warm-ups is 3,467.033 seconds. This excludes editing, conversion, grading, reporting and calendar gaps; it is not total human/project time or an efficiency claim.

One proposal was imported once (17.307 seconds). The original conversion-stage precheck failed before a converter process or candidate inference began. A separately recorded launcher repair preserved the virtual-environment interpreter; one manual continuation completed conversion preparation (29.880 seconds) and packaging (8.742 seconds), with no second import. Conversion retained its 1,800-second subprocess and 30-second precheck limits. These stage timings must not be added as independent nonoverlapping CPU costs. Editor model/version/effort/usage/billing were unobserved, so total editor tokens or monetary study cost cannot be supplied. Both the conversion-precheck failure and later supervisor interruption remain in the record.

## Interpretation, preservation and reproduction

The [saved-output review](../../docs/ifeval-v1-output-review.md) describes every held-out pair with a changed strict or loose instruction outcome (19 pairs), plus development item 17. Changes mix useful bounded/closed response structure, formatting tradeoffs, missing requested components and repetitive checker gains. This is descriptive evidence separate from official metrics; no substantive-quality score or post-hoc primary score was invented.

The original supervisor rejected position 95 without reconciling the durable predecessor completion. Saved logs and control flow support an event-snapshot/intent-file race; exact snapshot-read timing was not instrumented. No acknowledgement, reservation or model call existed for positions 95–128. The original candidate required forced termination; closure was verified but graceful unload was not. A separately versioned repair and continuation completed only those 34 positions with one additional warm-up. All original grades, ledgers, failures and reports remain unchanged. The candidate is complete across sessions, not a successful uninterrupted run.

Generation remained fresh conversations, thinking disabled, temperature 0, top-p 1, seed 42 reset per request, context 4,096, output cap 1,024, 120 seconds per request, WebLLM 0.2.86 and q4f16_1. Four injected empty-thinking IDs count against each cap but not sampled output tokens (at most 1,020 scored or 12 warm-up sampled tokens). Warm-ups never count toward compliance.

See [reproduction and provenance](../../docs/ifeval-v1-reproduction.md) for the actual transformation, source/deployment identities, shared-group quantization changes, dataset and checker pins, manual editor limitations, and reproduction scope; [ordered item IDs](item-ids.json); and [closeout accounting](result.json). Public aggregate consistency can be checked with `python3 scripts/verify_ifeval_closeout.py`. It does not regenerate, regrade or recompute frozen intervals. Original evidence verifiers additionally require private artifacts.

The [closeout verification](verification.json) records 66 passing offline tests on the staged source snapshot, 32 passing read-only continuation-evidence checks, and 53 unchanged snapshotted report/evidence/ledger files. The previously documented 300 unavailable historical files remain outside the preservation claim.

This remains an exploratory manual-editor study with unverified editor isolation and no verified causal interpretation of the chosen weight columns. Subset selection does not establish full-benchmark representativeness or absence of pretraining exposure. The 32 development and 128 completed test prompts are exposed for future search. No new inference, proposal, weight edit, conversion, or experimental frontier request was made for this closeout.

Upstream attribution remains with [Google Research IFEval (Zhou et al.)](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval), [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), Qwen, WebLLM/MLC, existing BrowserLLM work, Gufran's contribution and their preserved licenses.
