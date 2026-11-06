# Phase 4 verification

Checked 2026-11-06 with Python 3.13.5. Phase 4 implements benchmark data,
independent graders and scheduled paired analysis. **No Phase 4 model inference,
candidate search or benchmark experiment was executed. No accuracy improvement
is claimed.** Earlier Phase 3 smoke evidence is unchanged.

## Changes and scope

Pinned loaders retain source identities and exact bytes; preparation writes
explicit development, validation and final manifests. Dataset-specific
normalized-text and template-family audits keep selection families separate
from final data. MATH-500 and the original GSM8K test remain complete. IFEval's
66-prompt final subset is an explicit custom study partition, not the complete
official 541-prompt benchmark.

Independent grading uses Math-Verify 0.9.0 with pinned numerical/parser
libraries, exact numeric GSM8K grading, and the unchanged official IFEval
evaluator. Explicit policies separate reasoning from final output and freeze
truncation and retries. Diagnostic records remain separate from raw generations
and compact paired records. Production projection requires a complete frozen
item/sample schedule, retains every missing or failed case, and distinguishes
model failures from grader failures. Unresolved grader errors prevent an
identified semantic-effect claim.

## Checks actually executed

- The full default offline suite: **208 tests passed**, with no inference
  runtime installation or evaluator dependency required by those tests.
- **25 independently authored known-answer checks passed** through the actual
  pinned optional evaluators. They cover equivalent mathematics, absent and
  multiple answers, malformed output, truncation, reasoning boundaries,
  strict/loose instruction metrics, and evaluator failures. These are fixture
  checks, not model scores. See [known-answer-checks.json](known-answer-checks.json).
- Actual bounded benchmark acquisition: **11 files, 8,574,725 bytes**. All
  recorded source hashes and sizes verified. The real bundle was prepared,
  then checked by reconverting pinned cached bytes and comparing every item,
  source row, task projection, partition and audit.
- Official evaluator modules, Apache-2.0 license and pinned tokenizer resource
  were acquired and verified. The optional environment contains no model
  inference runtime. Dependency consistency checks passed in both the existing
  project environment and separate evaluator environment.
- CLI fixture coverage exercised freeze, grade, project, compare, append-only
  output rejection and missing-grade rejection. Failure/timeout and denominator
  cases are covered by the offline suite.
- Diff whitespace, Python compilation and the public-source publication audit
  passed. The publication audit ran from private temporary code outside source.

[build-checks.json](build-checks.json) records commands and observed check status;
[data-verification.json](data-verification.json) records data counts and digests;
[status.json](status.json) keeps evidence tracks separate. The public report
contains no withheld task text, raw benchmark generation, or final-data targets.
Prepared data and its detailed audit remain in ignored local artifacts.

## Prepared data

| Benchmark | Development | Validation | Final test |
| --- | ---: | ---: | ---: |
| MATH | 5,973 | 1,482 | 500 |
| GSM8K | 6,002 | 1,470 | 1,319 |
| IFEval custom partition | 377 | 98 | 66 |

Forty-four selection rows matched fixed-final text/template families; two
MATH training rows had empty source answers. They were quarantined with source
identities before any scheduling. Every source row is represented or explicitly
quarantined. No scheduled model failures are removed this way.

The original MATH archive returned HTTP 401. The pinned public EleutherAI
training mirror supplies shard/row identities; original filenames unavailable
in that mirror are not invented. This source choice and the custom IFEval
scope are explicit in [benchmark data](../../docs/benchmark-data.md).

## Reproduce

Use a fresh destination for each artifact. The commands below download only
the pinned dataset/evaluator assets; they run no model. Python 3.13.5 matches
the checked interpreter and grader identity. Other supported Python versions
produce a distinct grader identity and require their own validation.

```bash
python3.13 -m venv .local/phase04-env
.local/phase04-env/bin/python -m pip install -e '.[datasets,grading]'
.local/phase04-env/bin/python -m pip check
.local/phase04-env/bin/python -m unittest discover -s tests -v

.local/phase04-env/bin/python -m weight_space_lab acquire-benchmarks \
  --reference configs/benchmark_sources.json --out artifacts/benchmark-cache
.local/phase04-env/bin/python -m weight_space_lab prepare-benchmark-data \
  --reference configs/benchmark_sources.json --cache artifacts/benchmark-cache \
  --out artifacts/benchmark-data
.local/phase04-env/bin/python -m weight_space_lab check-benchmark-bundle \
  artifacts/benchmark-data --cache artifacts/benchmark-cache

.local/phase04-env/bin/python -m weight_space_lab acquire-ifeval \
  --out artifacts/official-ifeval
.local/phase04-env/bin/python -m weight_space_lab check-graders \
  --ifeval-root artifacts/official-ifeval \
  --out artifacts/known-answer-checks.json
```

For default lightweight-only reproduction, `python -m pip install -e .` and
`python -m unittest discover -s tests -v` are sufficient. Production artifact
commands and exact required inputs are documented in
[paired analysis](../../docs/paired-analysis.md#public-artifact-commands).

## Status and remaining prerequisites

| Track | Phase 4 status |
| --- | --- |
| Implementation/build | Complete; local checks passed |
| Fixture/evaluator validation | Passed; actual pinned evaluators used on independent fixtures |
| Real data preparation | Completed and verified against source bytes |
| Real-model grading | Not executed in this phase |
| Pilot/full experiment | Not executed; no scientific outcome |

The data audit detects its declared exact and skeleton families, not arbitrary
paraphrases or model-training contamination. Any additional semantic audit must
be decided before experiment freeze. Dataset terms remain upstream.

To produce real paired benchmark evidence, supply both arms' immutable
checkpoint and execution identities, explicit generation seeds, matched
prompt/decoding/precision settings, a complete frozen schedule and grading
policy, and real raw outputs. Qualify the chosen inference machine and runtime
under the existing native execution requirements. The native CLI is still
limited to development smoke; full benchmark execution and candidate search
remain later-phase work. Do not reinterpret the prior 16-token smoke as a
benchmark run. No hardware or editor service is needed for the offline checks
above; no paid infrastructure was provisioned.

Editor-development, evaluator-validation and evaluator-final bundles are
prepared. **Their directories do not enforce isolation.** Phase 7 must enforce
and verify storage, filesystem and tool access before an isolation claim.
Fresh-editor search additionally requires the actual Astra interface and model
identity, bounded budgets, and the candidate-selection protocol. None was
substituted or invented. The upstream browser benchmark was not modified.
