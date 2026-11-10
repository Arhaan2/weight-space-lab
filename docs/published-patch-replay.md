# Fixed published-patch evaluation

Phase 5 connects the verified p50c checkpoints, native generation runner,
independent graders, and paired analysis into an executable **published-patch
replay**. This label applies whether the replay improves, regresses, or differs
from the published result. It does not establish fresh-editor discovery.

The primary comparison is the unchanged checkpoint against reconstructed p50c
weights under identical inference, task, seed, stopping, and grading settings.
The original-to-identity comparison checks the independently verified byte-copy
and reload path. A separate temperature-1 comparison runs both unchanged and
reconstructed weights and is labeled **alternative decoding**. Its temperature
is fixed in advance; it is not a development-tuned result. No decoding setting
is selected by this workflow from pilot or final scores.

The complete replay result is **pending**. The preferred vLLM/CUDA runtime has
not been executed on the local machine. CPU development checks do not establish
GPU compatibility, long-context memory fit, or benchmark improvement.

## Scope and frozen settings

Preparation verifies the acquired base, saved reconstruction, independent
identity copy, prepared benchmark data, and cached source bytes before creating
execution inputs. The freeze binds complete checkpoint/file inventories,
source and dataset revisions, item identities and ordered sample schedules,
implementation and installed runtime identities, precision, tokenizer and chat
template, task wrapping, decoding and stopping, graders, output extraction,
truncation and retry policy, and comparison/analysis configuration. Changing
these requires a new freeze and new output destination; resume does not silently
adopt changed inputs.

The full primary schedule uses the published benchmark-specific repeat counts:

| Benchmark | Items | Samples per item | Generations per arm | Primary benchmark metric |
| --- | ---: | ---: | ---: | --- |
| MATH-500 | 500 | 4 | 2,000 | Average per-generation correctness |
| GSM8K test | 1,319 | 2 | 2,638 | Average per-generation correctness |
| IFEval full source | 541 | 2 | 1,082 | Strict prompt-level average per-generation correctness |

These counts and the thinking protocol are specified in the pinned
[published evaluation code](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/evaluate.py)
and [experiment description](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md).
The 541-prompt IFEval schedule restores the full pinned source for this replay;
it is different from Phase 4's 66-prompt custom final subset. Its reuse of
study development and validation prompts is explicit. This replay is not a
fresh held-out IFEval experiment.

The two primary arms use stochastic decoding with temperature 0.6, top-p 0.95,
top-k 20, generated-token presence penalty 1.5, thinking enabled, and a 32,768
new-token limit. The context limit is 34,816. The temperature-1 arms change
only temperature. Both checkpoints receive the same decoding configuration
within each paired comparison. Identity uses the primary decoding settings.
Full-stage requests have a 7,200-second hard timeout, model startup has a
300-second timeout, and each job has a total budget of 300 seconds plus
7,200 seconds per scheduled generation. There are no generation retries.
Timeouts can occur before the token cap and remain explicit failures in the
denominator; these operational limits are part of this replay's declaration.
MATH and GSM8K use the exact published math instruction after the problem;
IFEval receives its source prompt without that addition. See the pinned
[prompt builder](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/mathqa.py).

All five arms together schedule 28,600 full generations; the primary comparison
alone schedules 11,440. The maximum output-token workload is 937,164,800 tokens
for all five arms, or 374,865,920 for the primary pair. These are configured
upper bounds, not generated-token measurements. Actual token counts, missing
work, failures, and elapsed time remain visible in the report.

## Development validation, then freeze

Install and test the lightweight project without loading a checkpoint or
running any benchmark:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Use the [artifact workflow](model-artifacts.md),
[benchmark preparation](benchmark-data.md), and
[independent grading](independent-grading.md) to obtain the verified inputs.
Install only the intended inference backend in its own environment. Also
install the pinned dataset extra there, because preparation independently
reconstructs benchmark rows from verified source bytes on every freeze:

```bash
python -m pip install -e '.[datasets]'
```

Use a separate grading environment with the pinned evaluator dependencies
and the lightweight project installed for the `report-published-replay`
commands below. The project provides NumPy for paired analysis; installing only
the evaluator packages is insufficient. Match Python and NumPy to the frozen
analysis environment (NumPy 2.3.5 for the recorded CPU variation, 2.2.6 for the
pinned vLLM environment). Run preparation and generation
commands in the selected native inference environment. The
[CPU and GPU runtime environments](native-execution.md) have different
Transformers pins and must not be combined. No installation or test command
launches the replay.

A smoke uses one development example from each benchmark, one sample, and a
16-token cap per arm. It checks task wrapping, all checkpoint arms, raw evidence,
extraction, independent grading, and report production within a small budget.
The report also runs independent known-answer grader checks. A short truncated
response is a useful execution check but does not demonstrate a completed
answer or meaningful benchmark accuracy.

```bash
python -m weight_space_lab prepare-published-replay \
  --stage smoke --backend transformers \
  --root artifacts/published-patch-replay \
  --fresh-editor-workspace artifacts/fresh-editor-workspace \
  --acquisition artifacts/p50c-acquisition \
  --reconstruction artifacts/p50c-reconstruction \
  --identity artifacts/p50c-identity \
  --data artifacts/phase04-benchmark-data-verified \
  --cache artifacts/phase04-benchmark-cache

python -m weight_space_lab replay-smoke \
  --root artifacts/published-patch-replay \
  --acquisition artifacts/p50c-acquisition \
  --reconstruction artifacts/p50c-reconstruction \
  --identity artifacts/p50c-identity

python -m weight_space_lab report-published-replay \
  --root artifacts/published-patch-replay --stage smoke --report-id first \
  --ifeval-root artifacts/ifeval-evaluator
```

Use the actual verified IFEval evaluator directory for `--ifeval-root`.
Reports are written to
`artifacts/published-patch-replay/smoke/reports/first/report.json` and `README.md`.
Keep failed checks and earlier report revisions. Use a new `--report-id` when
repeating analysis; do not overwrite earlier evidence. Durable grade-cache
entries bind the exact raw record, grader, and policy and are reused, including
recorded evaluator failures. Repeating a report does not rerun generation or
choose a better result. A failed or pending smoke validation report exits with
status 2; a successfully produced full report may exit 0 while its measured
result is still explicitly pending. Review the smoke report
and resolve execution or evaluator failures before preparing a later stage.

The pilot uses development examples only: two items per benchmark, one sample
per item, and a 128-token cap. It is a bounded pipeline check and resource
measurement. The reduced pilot token cap is not the full reference decoding
budget, and pilot correctness is not a final replay score.

```bash
python -m weight_space_lab prepare-published-replay \
  --stage pilot --backend transformers \
  --root artifacts/published-patch-replay \
  --fresh-editor-workspace artifacts/fresh-editor-workspace \
  --acquisition artifacts/p50c-acquisition \
  --reconstruction artifacts/p50c-reconstruction \
  --identity artifacts/p50c-identity \
  --data artifacts/phase04-benchmark-data-verified \
  --cache artifacts/phase04-benchmark-cache \
  --development-report artifacts/published-patch-replay/smoke/reports/first/report.json

python -m weight_space_lab replay-pilot \
  --root artifacts/published-patch-replay \
  --acquisition artifacts/p50c-acquisition \
  --reconstruction artifacts/p50c-reconstruction \
  --identity artifacts/p50c-identity

python -m weight_space_lab report-published-replay \
  --root artifacts/published-patch-replay --stage pilot --report-id first \
  --ifeval-root artifacts/ifeval-evaluator
```

For vLLM, perform the smoke on the intended compatible CUDA machine using
`--backend vllm` and a separate replay root. Produce a matching development
report there before freezing its pilot or full stage. A Transformers smoke
cannot validate a different vLLM installation. Runtime changes require a new
freeze and a fresh smoke, not an implicit fallback.

## Explicit full run

The following commands are the expensive benchmark execution path. Run them
only after the matching backend's development validation, compute planning,
and full freeze are complete. Set `REPLAY_ROOT` to the evaluator-owned replay
root that contains that backend's successful smoke report.

```bash
REPLAY_ROOT=artifacts/published-patch-replay-vllm

python -m weight_space_lab prepare-published-replay \
  --stage full --backend vllm --root "$REPLAY_ROOT" \
  --fresh-editor-workspace artifacts/fresh-editor-workspace \
  --acquisition artifacts/p50c-acquisition \
  --reconstruction artifacts/p50c-reconstruction \
  --identity artifacts/p50c-identity \
  --data artifacts/phase04-benchmark-data-verified \
  --cache artifacts/phase04-benchmark-cache \
  --development-report "$REPLAY_ROOT/smoke/reports/first/report.json"

python -m weight_space_lab replay-full \
  --root "$REPLAY_ROOT" \
  --acquisition artifacts/p50c-acquisition \
  --reconstruction artifacts/p50c-reconstruction \
  --identity artifacts/p50c-identity

python -m weight_space_lab report-published-replay \
  --root "$REPLAY_ROOT" --stage full --report-id first \
  --ifeval-root artifacts/ifeval-evaluator
```

The `replay-smoke`, `replay-pilot`, and `replay-full` commands are distinct;
none promotes itself to a later stage. An interrupted run may use `--resume`
with the same command, root, checkpoint identities, and frozen environment.
Attempt history and failures remain in the durable journal. Resuming is not a
new opportunity to tune settings or replace the published patch.

## Report and interpretation

The primary endpoint is MATH-500 average per-generation correctness. The
schedule gives each item equal weight, averages that item's scheduled samples,
and keeps missing, failed, and timed-out samples in the denominator. With the
uniform repeat count this equals average correctness across scheduled
item/sample pairs. It is not best-of-N success, pass@N, or the fraction of
items with at least one correct generation.

The report preserves paired item-level uncertainty, full denominators,
missing records, execution errors and timeouts, grader errors and timeouts,
truncation and extraction failures, token use, runtime, and every improved and
regressed item. Bootstrap resampling retains all samples of an item together;
individual repeated generations are not treated as independent items. The
bootstrap seed and replicate count are frozen with the analysis. Partial runs
are labeled incomplete. A zero-filled operational result while work is
missing is not a completed measured replay.

GSM8K is a secondary correctness comparison. IFEval's primary reported measure
is strict prompt-level correctness; strict/loose and prompt/instruction-level
measures remain separately named. An instruction-weighted rate is distinct
from an equal-prompt average of instruction fractions. Temperature-1 results
are alternative decoding comparisons and cannot replace the declared primary
comparison after the results are known. Identity results remain visible even
when they reveal a discrepancy.

The [published unchanged summary](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/eval/unedited/merged.json)
and [published p50c summary](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/eval/p50c/merged.json)
are reference evidence, not results measured by this workflow. Preserve any
difference in score, item coverage, runtime, token use, or extraction outcome.
Do not modify the label or settings to make a replay appear to match them.

## Declared discrepancies

The artifact reconstruction is exact for the released sparse BF16 changes;
the following execution/evaluation differences remain explicit:

- The acquired checkpoint is the full conditional-generation release. Upstream
  used a text-only export. The complete file inventory and native active-tensor
  checks remain part of this project's evidence.
- This project pins vLLM 0.17.0 and its dependencies; upstream did not specify an
  installed vLLM version. The Transformers CPU/BF16 eager path is an additional
  explicit variation and is never substituted automatically.
- Upstream grouped repeated outputs with `n` and seed 0. This runner uses an
  explicit request and seed for each scheduled sample. Repeat counts match;
  identical random-number streams are not claimed.
- This project explicitly stops at EOS IDs 248044 and 248046 or the output cap,
  retains special tokens in raw evidence, and forbids silent prompt truncation.
  Upstream's implicit tokenizer/generation stopping metadata was not fully
  pinned.
- Phase 4's independent reasoning policy requires a closed `</think>` boundary
  before grading a thinking response. Upstream's mathematical extraction leaves
  the entire response available when that marker is absent. Both workflows
  strip through the last closing marker when it is present.
- GSM8K uses this project's declared numeric final-answer extraction and exact
  Decimal comparison, whereas the published harness applied its mathematical
  boxed-answer verifier. MATH extraction and malformed-output rules are also
  pinned to this project's independent grader. The official IFEval evaluator
  revision and dependencies are separately pinned here.
- Dataset revisions and the development split policy are this project's
  explicit identities. Full IFEval replay uses all 541 source prompts and is
  distinct from the custom held-out subset and from any fresh-editor study.

See [native execution](native-execution.md),
[independent grading](independent-grading.md), and
[paired analysis](paired-analysis.md) for detailed contracts. Grader failures
are unresolved correctness, even when an operational bound assigns zero credit.
Do not hide them as ordinary wrong answers or drop them from the denominator.

## Workload and pending compute

Use measured smoke output tokens and generation time to estimate the frozen
full schedule. The report distinguishes token-cap planning estimates from
observed tokens and time, and reports the assumptions. Include every primary,
identity, and alternative-decoding arm when planning the complete workload.

The earlier [three-arm CPU smoke](../reports/native-smoke/README.md) measured
0.723–0.760 output tokens per second on short 16-token responses. At that rate,
the full primary pair's output-token cap alone corresponds to approximately
5,707–6,000 serial days; all five arms correspond to approximately
14,267–15,000 days. These extrapolations are deliberately coarse. They assume
constant short-context throughput and exclude loading, grading, queueing,
retries, and interruptions; they are neither measured full-run runtimes nor
upper bounds on elapsed time. Long-context throughput and memory fit have not
been validated. Early EOS may reduce generated tokens substantially.

A suitable GPU development smoke and a new matching freeze are required before
using GPU performance estimates. If that compute is unavailable, preserve the
executable frozen workflow, development evidence, workload estimate, and an
explicit **measured full result pending** status. No accuracy improvement can
be inferred from successful reconstruction, installation, known-answer checks,
short smoke generations, or a forecast.

The [Phase 5 evidence report](../reports/published-patch-replay/README.md)
records a newer 15-generation development smoke, independent grading checks,
and the actual frozen CPU pilot/full inputs. That full freeze uses
`--smoke-measurement reports/published-patch-replay/smoke-performance.json`;
its 0.680–0.752 token/second observations replace the earlier reference for
that workload calculation. Both measurements remain short-context CPU
references and provide no GPU ETA.

## Separation from a later fresh-editor workspace

Keep the replay root outside the declared later fresh-editor workspace. The
workflow rejects overlapping resolved paths. Final replay tasks, targets,
raw generations, independent grades, per-item analyses, and full reports stay
in the evaluator replay root and must not be copied into later editor context.

This directory boundary is organizational protection, not enforced access
isolation. Phase 7 must enforce and test separate filesystem/tool identities
before any isolated fresh-editor claim. A later editor workspace should receive
only its approved development inputs and source code, without replay outputs or
final-test feedback. A future evaluation whose design was informed by this
replay requires a fresh held-out task set for the corresponding claim.
