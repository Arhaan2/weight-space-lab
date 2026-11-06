# Independent grades and paired analysis

Phase 4 adds `weight_space_lab.analysis_pipeline`. Its production entry point is
`compare_graded_runs`, which requires a complete frozen analysis manifest.
`compare_paired` remains the compact numerical primitive used by mechanics
fixtures; it cannot establish data completeness or provenance by itself.

## Freeze the denominator before execution

Call `build_analysis_manifest(split_manifest, items, benchmark=..., seeds=...,
policy=..., arms=..., stage=..., metric=..., evidence_kind=...)` using the
[prepared data bundle](benchmark-data.md), never the observed output list.
Every item of the selected benchmark in that split manifest must be present.
The builder expands the same explicit sample/seed list for every item and
binds canonical item hashes, source revision, source manifest, split manifest,
full schedule, grading policy, pinned grader, named metric, stage, and both
execution arms. IFEval items also retain their number of scheduled instructions.

Each arm declares `run_id`, `profile`, `configuration_sha256`, and `checkpoint`
with `model_id`, immutable `revision`, and content `sha256`. Supply the real
frozen execution configuration and checkpoint identities; do not use fixture
hashes as execution pins. Persist the returned manifest before starting either
arm. It has a content digest and accepts no implicit policy defaults.

The policy records reasoning/output separation, permitted terminal tokens,
truncation treatment, generation retries, evaluator timeout, and evaluator
retry limits. See [independent grading](independent-grading.md). A successful,
incorrect, malformed or truncated generation is never retried to obtain a more
favorable answer. The raw-record validator already enforces bounded retries
only after retryable execution errors or timeouts. Production projection also
checks that the raw history's retry policy equals the frozen declaration.

A manifest digest establishes integrity, not when it was created or whether
its declarations are truthful. Preserve the execution-side manifest and its
provenance. This phase does not enforce an editor access boundary; Phase 7
must enforce isolation outside the directory convention.

## Three separate schemas

1. Raw run records retain exact generation text, timing, token counts,
   checkpoint/configuration identity and every attempt.
2. Independent grade records retain the evaluator identity, exact raw-record
   hash, canonical item hash, policy hash, extracted answer, output/reasoning
   hashes, semantic result, metric components and evaluator diagnostics.
3. Compact records contain only `item_id`, `seed`, `status`, `correct`, and
   optional `score`. `project_arm` produces this exact schema alongside a
   separate diagnostic and accounting envelope. Passing either raw evidence
   or the envelope directly to `compare_paired` is rejected.

The projection rejects stale grades; changed targets, output, attempt history,
policy or grader identity invalidate the binding. It also rejects extra or
duplicate grades, records outside the schedule, changed sample/seed mappings,
fixture/real mixtures, mixed stages or splits, changed execution contexts, and
conflicting known checkpoint or dataset identities. Unavailable identities on
failed or missing model records remain null; this never removes the case.

`records.to_analysis_records` is a legacy low-level projection for externally
supplied compact grades. Real-model use now requires an explicit complete
benchmark item/sample schedule. It does not bind the Phase 4 independent grader
schema or both arms; production callers use `analysis_pipeline` instead. Its
separate fixture helper remains available for the original contract tests.

## Failure accounting

Every scheduled sample appears in each compact arm, including cases absent
from both raw arrays. Absent output becomes `missing`; observed model errors,
timeouts, and missing results retain their execution status and receive zero
credit. There is no fabricated grader invocation for failed model execution.

A completed generation must have an explicit independently produced grading
outcome. An absent grade is incomplete evaluation work and aborts analysis.
An observed evaluator exception or timeout projects to compact `error` and
zero credit, with a separate `grader_error` or `grader_timeout` category. The
original generation status is preserved in diagnostics and generation counts.
These cases are **not** labeled semantically incorrect model answers.

Unresolved evaluator failures make `semantic_grading_complete` false. Each arm's
zero-credit score is then a conservative bound, and the paired delta is not an
identified semantic effect; the difference of two lower bounds is not itself a
lower bound on the true effect. Resolve grading failures under a new declared
and auditable evaluation version before making a correctness claim. Cases are
never dropped from the denominator to improve a score.

Truncation policy `grade` honors a returned semantic answer; `fail` assigns zero
credit while retaining `truncated` status. Complete-but-incorrect, truncated,
missing and evaluator-failed cases therefore remain distinguishable. Usable
text, valid JSON, or normal process completion supplies no semantic grade.

## Named metrics and weighting

MATH and GSM8K project independently verified `accuracy`. The primary paired
estimator averages scheduled samples within each item, then weights items
equally. Confidence intervals resample complete items with all their repeats;
repeats are not treated as independent benchmark questions.

IFEval freezes one of four explicitly named metrics:

- `prompt_level_strict_accuracy`
- `prompt_level_loose_accuracy`
- `instruction_level_strict_accuracy`
- `instruction_level_loose_accuracy`

The projection reports all four official aggregation forms. Prompt-level
numerators count prompts satisfying every instruction, with every scheduled
prompt in the denominator. Instruction-level numerators sum compliant
instructions, with every scheduled instruction in the denominator, including
instructions attached to missing or failed generations.

For an instruction-level paired analysis, the compact `score` is that prompt's
fraction of compliant instructions. The paired estimator therefore measures
**equal prompt mean instruction compliance**, which can differ from the
official instruction-weighted aggregate. Both are reported with those labels;
the item-clustered interval is not relabeled as an interval for the official
instruction-weighted statistic. For example, a one-instruction prompt passing
and a three-instruction prompt failing yield an equal-prompt mean of 0.5 but
an official instruction accuracy of 0.25.

Fixture, smoke, pilot and full analyses preserve their evidence and stage
labels. A projection or descriptive comparison alone does not establish an
executed experiment or language-model improvement. No benchmark experiment is
part of Phase 4.

## Local checks

```bash
python -m unittest discover -s tests -p test_analysis_pipeline.py -v
python -m unittest discover -s tests -p test_records.py -v
python -m unittest discover -s tests -p test_evaluation.py -v
```

The pipeline cases use independent hand-authored questions, answers and metric
components. They test schema binding and failure denominators, not model scores.

## Public artifact commands

After preparing a bundle, supply `seeds.json` as an explicit array of unique
nonnegative seeds and `arms.json` with the two real frozen arm identities
specified above. These execution identities remain a prerequisite; this
project does not ship fabricated production hashes. Select a named IFEval
metric explicitly instead of the mathematics `accuracy` metric.

For example, the following commands apply to an already frozen full MATH
execution. The execution records and arms file must come from that execution;
these commands do not run inference or create the missing prerequisites:

```bash
python -m weight_space_lab freeze-analysis \
  --split-manifest artifacts/benchmark-data/evaluator-final/manifest.json \
  --items artifacts/benchmark-data/evaluator-final/items.json \
  --benchmark math --seeds artifacts/run-inputs/seeds.json \
  --policy configs/grading-policy.json --arms artifacts/run-inputs/arms.json \
  --stage full --metric accuracy --evidence-kind real_model \
  --out artifacts/run-inputs/analysis-manifest.json

python -m weight_space_lab grade-records \
  --records artifacts/baseline/records.json \
  --items artifacts/benchmark-data/evaluator-final/items.json \
  --policy configs/grading-policy.json \
  --out artifacts/grades/baseline.json
python -m weight_space_lab grade-records \
  --records artifacts/candidate/records.json \
  --items artifacts/benchmark-data/evaluator-final/items.json \
  --policy configs/grading-policy.json \
  --out artifacts/grades/candidate.json

python -m weight_space_lab compare-graded \
  --manifest artifacts/run-inputs/analysis-manifest.json \
  --baseline-records artifacts/baseline/records.json \
  --candidate-records artifacts/candidate/records.json \
  --baseline-grades artifacts/grades/baseline.json \
  --candidate-grades artifacts/grades/candidate.json \
  --bootstrap-samples 10000 --seed 0 \
  --out artifacts/analysis/comparison.json
```

Create and persist the analysis manifest **before** generating the records.
The example's bootstrap seed controls numerical resampling only; it does not
choose generation seeds. Each raw array must contain only the selected arm
and benchmark. For IFEval grading, additionally pass
`--ifeval-root artifacts/official-ifeval` to `grade-records` and predeclare one
of the four IFEval metric names at freeze time. Run all these operations with
the same pinned evaluator environment and Python patch version because the
grader identity includes that interpreter version and local adapter source.

A grading CLI invocation writes diagnostics and exits with status 2 if a
grader errors or times out. Keep that artifact and resolve the failure under
a new auditable grading run; never overwrite an earlier diagnostic. The
comparison retains explicit grader failures in its denominator and labels
the unresolved interpretation. A missing grade stops comparison entirely.
