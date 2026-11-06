# Benchmark data and split manifests

Phase 4 acquires exact benchmark source bytes and prepares explicit development,
validation and final-test manifests before any experiment. The source catalogue
is [configs/benchmark_sources.json](../configs/benchmark_sources.json): every
file has an observed immutable revision, SHA256, byte size, and row count.
Downloads are bounded and the cache becomes visible only after all files verify.
No inference framework or dataset download is needed for the default tests.
The optional MATH parquet reader is pinned to `pyarrow==19.0.1`.

## Sources and identities

| Dataset use | Pinned source | Identity preserved |
| --- | --- | --- |
| MATH development/validation pool | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3), seven original training split parquet shards | Exact shard-relative filename and zero-based row offset, plus source file hash |
| MATH final test | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be) | Original `unique_id`, including original subject filename |
| GSM8K | [OpenAI source](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c) | Original split and zero-based line position, bound to exact JSONL bytes |
| IFEval | [Google Research source](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval) | Original integer `key` represented as a decimal string; instruction IDs and arguments unchanged |

The original `hendrycks/competition_math` archive at revision
`71b758ecc688b2822d07ffa7f8393299f1dc7cac` returned HTTP 401 during preparation.
The explicitly named EleutherAI mirror provides the original 7,500 training
problems, not generated substitutes. Its parquet schema does not preserve the
original competition filenames. Those unavailable identities are not invented:
the stable source identity is the pinned shard and row position. MATH-500 keeps
its original filenames. Normalized text and template-family checks bridge the
different representations; an original filename correspondence is not claimed.

Every canonical item includes `benchmark`, `item_id`, `dataset_revision`,
`split`, exact `task_text`, `target`, `source`, and `metadata`. The target is an
explicit answer for MATH/GSM8K, or the official `instruction_id_list` and
`kwargs` for IFEval. Source solutions, file hashes and row positions remain
available for evaluator audit. Canonical JSON hashes bind the entire item,
including the target and assigned split. Revision, source identity, or target
changes therefore require a new manifest.

MATH source-answer extraction reads the last balanced `boxed`/`fbox` argument,
including nested braces and escaped literal braces. The unambiguous one-digit
unbraced source form is supported. Empty or malformed final source answers are
never guessed or recovered from an earlier box. Such rows may be quarantined
only from the selection pool, before scheduling, with source identities and
reasons retained. An invalid fixed final source answer stops preparation.
GSM8K source targets come from the final `####` delimiter. These are source
conversion rules; model-output scoring is a separate independent operation.

## Frozen split and overlap policy

The `template-groups-v1` policy is deterministic and independent of model
outputs. The fixed MATH-500 and original GSM8K test sets remain complete.
Training rows that match fixed final identities, normalized text, or a template
skeleton are quarantined from selection. The remaining whole template families
are assigned to validation when the integer family hash modulo five is zero,
and otherwise development. Thus nominal 80/20 allocation applies to families;
item counts need not be exactly 80/20.

The audit uses Unicode NFKC, case folding and collapsed whitespace for exact
text comparison. Its additional dataset-specific family checks are:

- **MATH:** replace numeric literals and single-letter variables with markers,
  preserving surrounding mathematical language and LaTeX structure. This catches
  coefficient/variable substitutions in the same textual template.
- **GSM8K:** replace numeric literals and capitalized name-like words with
  markers. Matching skeletons catch common number/name substitutions; this is
  deliberately conservative and can group unrelated short wording.
- **IFEval:** group by the sorted multiset of official instruction IDs,
  preserving multiplicity. Every prompt with the same instruction combination
  stays in one split. Individual instruction types may appear in multiple
  combinations across splits: this is a held-out combination design, not a
  claim that every instruction type is unseen.

IFEval's official 541 prompts have no independent official development split
in this source. This study explicitly partitions their instruction families:
hash modulo five zero is final test, one is validation, and two through four
are development. **The resulting 66-prompt final subset is study-specific and
must not be reported as the full official 541-prompt IFEval benchmark.** The
full pinned source remains in the evaluator cache for a separately declared
replay study. Changing this partition or evaluating the full source requires
a different predeclared study manifest.

These checks are a documented template-family audit, not a semantic duplicate
oracle. Paraphrases, equivalent problems expressed in different prose, diagrams,
and model-training contamination can remain undetected. Any later manual review
or broader semantic audit must occur before freezing an experiment and record
its decisions; it must not use favorable model scores to choose exclusions.

The checked preparation produced these data counts; they are not accuracy scores:

| Benchmark | Development | Validation | Final test |
| --- | ---: | ---: | ---: |
| MATH | 5,973 | 1,482 | 500 |
| GSM8K | 6,002 | 1,470 | 1,319 |
| IFEval study-specific partition | 377 | 98 | 66 |

Forty-four selection rows matched fixed-final text/template families. Two MATH
training rows had empty source answer boxes and were separately quarantined.
Every source row is accounted for by an item or an explicit quarantine record.
These are pre-experiment data decisions; scheduled missing generations,
timeouts, grader failures and incorrect outputs are never quarantined from the
analysis denominator.

## Audience bundles and verification

Preparation writes three explicit manifests, each with its items and generation
tasks:

```text
benchmark-bundle/
  editor-development/{manifest,items,tasks}.json
  evaluator-validation/{manifest,items,tasks}.json
  evaluator-final/{manifest,items,tasks}.json
  source-reference.json
  overlap-audit.json
  bundle.json
```

`tasks.json` is the explicit projection accepted by the native runner: one user
message containing exactly `task_text`. It contains no target answers. No answer
instruction, reasoning prefix, chat wrapper, or other prompt transformation is
silently added. If a later protocol needs one, it must declare and hash that
transformation before generation. The current native CLI remains restricted to
its development smoke scope; writing task files does not authorize a full run.

`check-benchmark-bundle` verifies internal hashes, all source-file identities,
source-row coverage, exact task projection, deterministic role/split allocation,
and a freshly calculated overlap audit. `--cache` additionally verifies pinned
raw bytes and replays conversion and partitioning against them. Internal hashes
alone establish consistency, not independently authenticated provenance.

**Directory names do not enforce isolation.** The operator preparing these
bundles can read all of them. The editor should receive only the development
bundle; validation feedback must follow the declared selection procedure and
final data must remain evaluator-only. Phase 7 must enforce the filesystem,
tool and artifact access boundary before the fresh-editor experiment. Do not
give the editor the parent bundle directory, full source cache, or overlap audit.

## Reproduction

Run from the repository root with Python 3.11 or later. Output directories must
be new; evidence is not overwritten. Keep source data and prepared bundles in
the ignored `artifacts` tree or another evaluator-controlled storage location.

```bash
python -m pip install -e '.[datasets]'
python -m weight_space_lab acquire-benchmarks \
  --reference configs/benchmark_sources.json --out artifacts/benchmark-cache
python -m weight_space_lab prepare-benchmark-data \
  --reference configs/benchmark_sources.json --cache artifacts/benchmark-cache \
  --out artifacts/benchmark-bundle
python -m weight_space_lab check-benchmark-bundle artifacts/benchmark-bundle \
  --cache artifacts/benchmark-cache
python -m unittest discover -s tests -p 'test_benchmark_data.py' -v
```

Acquisition needs public HTTPS access and at most 12 MB of downloaded source
data by default. Loading the seven MATH parquet shards additionally needs the
pinned optional reader. Bundle checks without `--cache` and all fixture tests
use only the project's ordinary dependencies. No GPU, model, editor connection,
paid service, or model generation is needed to prepare the data.

## Attribution

MATH is due to Dan Hendrycks and collaborators; see [Measuring Mathematical
Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874). The
mirror and MATH-500 packaging retain EleutherAI and HuggingFaceH4 provenance.
GSM8K is due to Karl Cobbe and collaborators at OpenAI; see [Training Verifiers
to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) and the upstream
[MIT license](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE).
IFEval is due to Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma,
Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou; see [Instruction-Following Evaluation
for Large Language Models](https://arxiv.org/abs/2311.07911) and Google Research's
[Apache-2.0 license](https://github.com/google-research/google-research/blob/e6890f85757dd84e27ca6df2dd30651dafad28e0/LICENSE).
Original dataset terms remain upstream; this project's MIT license does not
relicense source datasets. The catalogue and audience bundles retain these
source identifiers, revisions and attribution notes.
