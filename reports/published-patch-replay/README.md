# Published-patch replay — Phase 5

**The full measured replay result is pending. No full benchmark or pilot was
launched.** The executable workflow joins verified checkpoints, pinned data,
native generation, independent grading, and paired item-level analysis.

The primary comparison is unchanged versus reconstructed p50c weights under
identical settings. Identity-copy and temperature-1 controls are predeclared.
Temperature 1 is an alternative decoding comparison; no settings are selected
using test results. Results remain labeled published-patch replay regardless
of agreement with the published improvement.

## Execution and evidence

The [workflow instructions](../../docs/published-patch-replay.md) provide
separate preparation, `replay-smoke`, `replay-pilot`, `replay-full`, and reporting
commands. Installation, wheel building, and the default test suite never run
the full benchmark. The [build checks](build-checks.json) record 255 passing
offline tests and successful wheel construction.

The final development smoke completed **15/15 generations**, with 240 output
tokens and 378.17 seconds of accounted execution time (335.58 seconds of
generation time). All 15 reached the 16-token cap; there were no missing
generations, execution errors, timeouts, or grader errors. All 25 independent
known-answer checks passed. Original and identity-copy token sequences matched
on each of the three development items.

Under the frozen extraction policy, all 15 truncated responses were incorrect.
Each primary arm therefore scored 0/1 on each development benchmark item;
paired deltas were zero with degenerate single-item intervals [0, 0]. There
were no improved or regressed smoke items. These are short pipeline checks,
not estimates of benchmark accuracy. The [aggregate report](report.json)
preserves correctness, uncertainty, denominators, missing/error/truncation
counts, token use, runtime, and the pending full results.

Pilot and full inputs were frozen only after this validation passed. The
[freeze summary](freeze-summary.json) records their identities and settings;
the complete local full freeze is
`artifacts/published-patch-replay/full/inputs/study.json`. Its pending report is
`artifacts/published-patch-replay/full/reports/pending/report.json`, with a
readable `README.md` beside it. Neither stage has any generation attempts.
The freeze is the explicit CPU variation; it does not validate vLLM.

The first report invocation stopped before grading because the separate
evaluator environment lacked the project's NumPy analysis dependency. NumPy
2.3.5 was installed to match the frozen native environment, and reporting
then passed without repeating generation. This prerequisite failure and the
native execution logs remain in the local replay command logs.

An initial development trial was stopped by the source-drift guard after a
reporting correction: attempts that all fail to generate must leave measured
correctness pending. Its 11 completed generations (176 output tokens), frozen
inputs, journals, and matching source wheel are preserved separately. These
ungraded preliminary attempts are not included in the final smoke's scores or
resource totals. See [the preliminary trial record](pre-freeze-development.json).

## Full workload and compute limit

| Benchmark | Items | Samples/item | Generations/arm |
| --- | ---: | ---: | ---: |
| MATH-500 | 500 | 4 | 2,000 |
| GSM8K | 1,319 | 2 | 2,638 |
| Official IFEval source | 541 | 2 | 1,082 |

All five arms schedule 28,600 generations and up to 937,164,800 output tokens.
The primary pair alone schedules 11,440 generations and 374,865,920 output
tokens. Every scheduled item and sample remains in the denominator, including
mutually absent records, errors, timeouts, and truncations.

The [new measured development smoke](smoke-performance.json) produced
0.680–0.752 output tokens/second. The full freeze uses this measurement through
`--smoke-measurement reports/published-patch-replay/smoke-performance.json`.
At those short-context rates, reaching every token cap would require roughly
14,428–15,940 serial generation days for all five arms, excluding loading,
grading, and retries. This is a measured-hardware planning reference, not a runtime
upper bound or a GPU ETA. Long-context performance and memory fit are unknown.
The local machine has no configured CUDA/vLLM execution path; practical full
compute is unavailable. The preferred GPU variation requires its own measured
development validation and matching freeze.

## Interpretation and separation

The full report retains null measured correctness and uncertainty while the
run is pending. It also preserves scheduled denominators and missing counts,
instead of presenting unexecuted cases as measured zero accuracy. For observed
runs, correctness is the average over scheduled generations, **never best-of-N
success**. Paired bootstrap intervals resample whole source items; all repeated
samples stay together. Identity and alternative decoding results stay separate.

The complete frozen inputs, raw records, grades, and item-level reports stay
under the ignored replay artifact root. The reserved fresh-editor workspace
is disjoint and receives no replay outputs. Runtime path checks reject nesting
and symlink aliases. This organizational separation does not establish the
later Phase 7 filesystem/tool access boundary. Public files here contain
status and aggregate development evidence, not final replay outputs.

Preserved deviations include the full conditional-generation checkpoint in
place of the upstream text-only export; the local Transformers CPU/BF16
variation; newly pinned runtime/evaluator versions; separate per-sample seeds;
explicit EOS handling; the independent thinking/extraction and GSM8K numeric
grading policies; and reuse of all 541 IFEval prompts, including development
prompts. The last point makes this a full-source replay, not a fresh held-out
IFEval study. Full results and improved/regressed items remain pending.
