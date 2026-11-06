# Versioned execution evidence

`weight_space_lab.records` defines execution evidence independently of the
compact, externally graded records accepted by `compare`. Neither schema
validates the scientific truth of an output or a grade. Phase 1 supplies the
contracts and CPU fixtures. Phase 3 adds the [native runner](native-execution.md);
Phase 4 adds [independent graders](independent-grading.md) and a [manifest-bound analysis pipeline](paired-analysis.md).

## Run records, version 1

Use a JSON array of run records with exactly the fields below. Unknown fields
are rejected. `validate_run_record` validates one record and
`validate_run_records` also rejects duplicate run/benchmark/item sample indexes
or seeds. These functions return independent copies. Acquisition and checkpoint
reconstruction produce artifact audits rather than generation records.

| Field | Contract |
| --- | --- |
| `schema_version`, `record_type` | Integer `1`, string `run` |
| `run_id` | Nonempty identifier for one execution arm |
| `evidence_kind` | `fixture` or `real_model`; synthetic evidence must remain labeled `fixture` |
| `profile` | `native_development_smoke`, `fixed_patch_evaluation`, `fresh_editor_search`, `frozen_final_evaluation`, or `browser_evaluation` |
| `stage` | `smoke`, `pilot`, or `full` |
| `benchmark`, `item_id` | Nonempty identifiers; benchmark names scope item identities |
| `dataset_revision` | Immutable 40- or 64-character lowercase revision/content digest; null is permitted for uncompleted work |
| `split` | `development`, `validation`, or `test` |
| `sample_index`, `seed` | Nonnegative integers; booleans are rejected |
| `checkpoint` | Exactly `model_id`, `revision`, and `sha256`; completed outputs require all three. Revision is an immutable revision/content digest, SHA256 identifies the actual checkpoint. Unavailable values in failed/missing work remain null. |
| `configuration_sha256` | SHA256 of the full execution configuration, including decoding, precision, protocol provenance, and retry policy |
| `status` | `ok`, `truncated`, `error`, `timeout`, or `missing` |
| `raw_output` | Exact ungraded generated text, including partial text, or null when unavailable |
| `finish_reason` | `stop`/`eos` for `ok`; `length` for `truncated`; `error`, `timeout`, or `not_started` for the corresponding status |
| `token_counts` | Exactly `input`, `output`, `total`; each is a measured nonnegative integer or null when unavailable. When all are measured, total must equal input plus output. Total cannot be smaller than any known component. |
| `duration_seconds` | Finite measured nonnegative duration or null when unavailable; required for `ok`/`truncated`. This is terminal-attempt duration, not the total retry budget. |
| `error` | Null, or exactly `code` and boolean `retryable`. Codes are stable lowercase identifiers, never raw provider exceptions or stack traces. |
| `retry_policy` | Exactly `mode` (`none` or `retry_errors`) and positive `max_attempts`; `none` requires one attempt |
| `attempts` | Ordered list of attempt objects, each containing `attempt_index` and every outcome field from `status` through `error` |

A missing result has no attempts and null output, error, token measurements and
duration. It is not an executed zero-token generation. An attempted failure
belongs under `error` or `timeout`, even when it produced no output. Completed
records require identified checkpoint and dataset, raw output (which may be an
empty string), measured duration, and no error. Unreported token counts remain
null rather than invented zeros. A checkpoint content hash does not replace its
source revision or establish that the source was actually verified.

Smoke records use development items. Fresh-editor search cannot use final-test
items. Frozen final evaluation requires the `full` stage and `test` split.
Benchmarks may have different sampling schedules; `sample_index` and `seed`
must retain the predeclared benchmark schedule. Records do not infer the absent
schedule: keep it alongside the evidence so the analysis can account for every
scheduled missing result.

Attempt indexes start at zero and are contiguous. The record outcome must
exactly equal its last attempt. Only a failure explicitly marked retryable can
precede another attempt, within the predeclared maximum. Completed or truncated
outputs cannot be retried under this schema. This prevents selecting the best
successful response after seeing answers. Retain all attempt output, time and
token measurements when counting the search budget. Unknown measurements stay
unknown; summing only known values is a lower bound, not complete usage.

The validator checks the declared policy and history. An eventual runner must
enforce these limits before calls, persist attempts as they happen, bind the
configuration digest to the actual frozen configuration, and ensure that no
attempts were omitted. Validation cannot prove an externally supplied record
is honest. Raw output may contain sensitive or restricted task material: review
it before publication, keep private errors outside source, and preserve any
required task or model licensing.

## Independent grading and compact analysis

The compact analysis contract remains unchanged: `item_id`, `seed`, `status`,
boolean `correct`, and optional `score`. Do not add raw evidence fields to that
format or use completed generation as evidence of correctness.

`to_analysis_records(records, grades, benchmark=..., run_id=..., schedule=...)` explicitly
rejects fixture evidence. The separate `to_fixture_analysis_records` helper
retains `{evidence_kind: "fixture", scientific_result: false, records: ...}`;
this envelope is not a scientific comparison input. Scientific projection
selects one run and benchmark. It requires one execution context, including
configuration, checkpoint, dataset, split, stage and fixture/real classification.
Keep smoke, pilot and full runs in separate analyses.
All grades in one projection must use the same immutable grader revision.
Unknown dataset or checkpoint identity on a failed/missing record does not
discard that failure: the projection rejects conflicting known identities and
leaves unknown values null in the raw evidence. Observed sample indexes and
seeds must have a consistent one-to-one mapping across the selected benchmark;
the supplied complete benchmark schedule additionally preserves absent samples. The production pipeline binds that schedule to all items in a frozen split manifest.

Each independently supplied grade has `item_id`, `seed`, boolean `correct`,
immutable `grader_revision`, and `run_record_sha256`, plus optional `score` in
`[0, 1]`. Compute the raw evidence identity with `run_record_sha256(record)`.
This binds a grade to the exact output and attempt history. Every completed or
truncated result requires a matching grade. Error, timeout and missing records
project to failure, without a fabricated grader call. Duplicate, extra,
unmatched or stale grades are rejected. The compact projection does not contain
the source evidence: preserve raw records, grades and schedules alongside it.

The legacy grade-binding helper above accepts externally prepared compact grades. New benchmark work uses the richer versioned diagnostic grade schema and frozen policy in [independent grading](independent-grading.md), then the explicit projection in [paired analysis](paired-analysis.md). Keep grader errors separate from model execution failures. Both remain in scheduled denominators; an unresolved grader error prevents a complete scientific correctness claim.

## Independent progress status, version 1

`empty_status_record()` starts every track at `not_started`.
`validate_status_record` checks a record with `schema_version: 1`,
`record_type: "status"`, and exactly four independent tracks:

| Track | Completion status | Required evidence kind |
| --- | --- | --- |
| `build` | `passed` | `build_log` |
| `fixture_validation` | `passed` | `fixture_report` |
| `real_model_validation` | `passed` | `real_model_report` |
| `experiment` | `completed` | `experiment_report` |

Every track has `status`, `evidence`, `reason`, and `outcome`. Other statuses are
`not_started`, `in_progress`, `failed`, and `blocked`. Evidence is a list of
objects containing exactly its track's `kind` and a `sha256` content digest.
Completed/passed and failed checks require nonempty evidence. Blocked or failed
tracks require a reason; an unexecuted track has no evidence. A completed
experiment requires `outcome` to be `positive`, `negative`, or `inconclusive`.
All other outcomes remain null. A negative result is a completed experiment.

Passing fixture validation never advances real-model validation or experiment
status. Fixture evidence cannot satisfy a real-model evidence reference.
References identify reports, not a trust guarantee: their contents and actual
execution must still be checked before publication. Do not fabricate a report
digest to make an unexecuted track pass.

CPU-only contract checks:

```bash
python -m unittest discover -s tests -p test_records.py -v
```
