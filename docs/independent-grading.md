# Independent answer grading

Phase 4 adds deterministic answer grading, separate from inference and from
compact paired-analysis records. The graders never ask the experimental editor
to judge itself. Known-answer fixtures validate evaluator behavior; they do not
measure a language model. No benchmark experiment or fresh-editor search is
performed by these commands.

## Pinned evaluators

- MATH uses [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify)
  release **0.9.0**, `latex2sympy2_extended==1.11.0`, `sympy==1.14.0`,
  `antlr4-python3-runtime==4.13.2` and `mpmath==1.3.0`. Parsing uses only the
  selected final expression, LaTeX extraction with no string fallback, and
  strict normalization (`malformed_operators=False`, `nits=False`). Verification
  uses strict comparison, six-place float rounding, 15-digit numerical
  precision, and no set/relation relaxation. Equivalent expressions can pass;
  a valid JSON object or merely parseable output cannot establish correctness.
- GSM8K uses exact decimal equality after the extraction policy below. It
  follows the dataset's `####` answer convention, with a declared final-box
  alternative for mathematical-model output. It does not use a model judge.
- IFEval calls the **unmodified official** Google Research
  [strict and loose evaluators](https://github.com/google-research/google-research/blob/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval/evaluation_lib.py)
  at `e6890f85757dd84e27ca6df2dd30651dafad28e0`. Acquisition downloads four
  source modules and their Apache-2.0 license, verifies every SHA256, and retains
  their upstream authorship. It does not vendor or rewrite the evaluator.
  All Python dependencies are pinned in `configs/grading-requirements.txt`.
  The English NLTK sentence data comes from the pinned `punkt_tab.zip` at
  [nltk_data revision 4f15a3d89eefe9748ec1c05be495d91289197155](https://github.com/nltk/nltk_data/tree/4f15a3d89eefe9748ec1c05be495d91289197155),
  SHA256 `e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106`.
  The original archive, including upstream metadata, remains in the asset
  bundle. Grading neither downloads data nor substitutes tokenizers. Python's
  random generator and langdetect use seed 0 independently of generation seeds.

`grader_identity(benchmark)` hashes the adapter and worker source, evaluator
pins and the exact Python interpreter version. IFEval also binds official
source hashes, tokenizer data and evaluator randomness. Freeze this identity
in the analysis manifest before generating outputs. Check, grade and project
with the same Python version; a different adapter or interpreter is a new
evaluation condition. Exact package-version checks fail closed. Source files,
resources, unexpected files and symlinks are checked before official code is
imported. The dependency environment is trusted installed software, not a
sandbox for arbitrary evaluator code.

## Output and final-answer policies

`configs/grading-policy.json` is an explicit proposed policy for Qwen thinking
output. It is not a claim that an experiment has been frozen. Each analysis
manifest binds the complete selected policy and its digest. Reasoning is
excluded identically across MATH, GSM8K and IFEval:

- `plain` grades the full generated response.
- `think_tags` accepts either no think block or one complete leading
  `<think>...</think>` block; only text after the block is graded.
- `qwen_thinking` requires exactly one `</think>` delimiter and permits the
  opening `<think>` token to have been supplied by the prompt. Only text after
  the delimiter is graded. An unclosed thinking response has no final answer.

Only explicitly listed trailing `<|im_end|>` and `<|endoftext|>` tokens are
removed. No broad special-token filter, Markdown cleanup, JSON extraction,
whitespace rewriting, or search for an answer inside reasoning occurs.
Malformed boundaries produce an incorrect model response with a diagnostic
code. Final-response and reasoning digests are recorded alongside the raw
record's digest; raw text remains in execution evidence.

MATH selects the **last balanced `\boxed{...}` expression**. Earlier boxes are
never selected because they happen to match. A missing box, empty box,
unclosed box, nested boxes (unsupported), or parser rejection is an incorrect
response. Nested boxes cannot cause an inner expression to replace the full
final expression. The complete
selected expression is passed to Math-Verify; a symbolic parser/verifier
decides mathematical equivalence. An unparseable reference answer is a grader
failure, not a model error. This is an answer-equivalence metric: it does not
verify the proof or reasoning and can accept algebraically equivalent forms
even when a problem requests a particular presentation.

GSM8K selects the text after the **last `####` marker**, which must consist
entirely of one signed decimal numeral (optional correctly grouped thousands
commas). If no marker exists, it uses the last balanced box under the same
rule. It accepts `1,200.00` as equal to `1200`; rejects units, prose, exponents,
fractions, malformed comma grouping and nonfinite values; and never falls
back to the last number in a chain of reasoning. These rules require matching
prompt instructions when an experiment is frozen. They do not silently alter
generation prompts or decoding.

IFEval receives the separated final response and original instruction IDs and
arguments. Its outputs are labeled explicitly:

| Field | Meaning |
| --- | --- |
| `prompt_level_strict_accuracy` | All instructions pass strict evaluation for this response |
| `prompt_level_loose_accuracy` | All instructions pass official loose evaluation for this response |
| `instruction_level_strict_accuracy` | Strict instructions passed / instructions scheduled for this response |
| `instruction_level_loose_accuracy` | Loose instructions passed / instructions scheduled for this response |
| `instruction_strict`, `instruction_loose` | Individual official instruction outcomes |

The default `correct` field is **prompt-level strict**. Report instruction-level
corpus accuracy as total passed instructions divided by total scheduled
instructions, not the unweighted mean of prompt ratios. Preserve all four
metrics and their instruction counts. Valid JSON can satisfy a JSON-format
instruction, but it cannot satisfy an additional missing keyword or establish
factual correctness. IFEval measures its specified instruction constraints.

## Truncation, retries and evaluator failures

Every policy states `truncation_policy` before execution. `grade` honors an
answer already present in truncated final output; `fail` forces it to zero.
Completion-within-budget remains a separate metric and always fails a
truncated generation. No reasoning-only output receives a final-answer grade.

Generation retries must match the raw execution record's declared bounded
`none` or `retry_errors` policy. Successful, incorrect, malformed and truncated
answers are never retried for a better grade. All earlier attempts remain in
raw evidence. The final attempt is the grading input.

Each optional evaluator runs in a fresh subprocess with a predeclared hard
wall-clock deadline (1–120 seconds, including import). It has **one grading
attempt**. Timeouts, missing or mismatched dependencies, invalid references,
official evaluator exceptions and invalid worker responses produce explicit
`grader_timeout` or `grader_error` records with null correctness. They never
become successful answers or disappear from the schedule. Raw model failures
produce `not_applicable` grades with the model-failure diagnostic.

Fix an evaluator problem in a separately versioned grading run over the same
immutable raw generations. Keep the previous diagnostics. Do not regenerate
model answers to recover a grading failure. Production analysis retains these
cases in the denominator and identifies unresolved grading as an inconclusive
correctness estimate; its zero-assigned aggregate is a bound, not an estimate
that labels a broken grader as a model error.

## Diagnostic contract and local verification

`grade_record(record, item, policy)` produces a versioned diagnostic record
bound to the complete raw record, canonical item including target, grading
policy and grader identity. It contains statuses, extracted answers and IFEval
instruction vectors, without duplicating raw generations. `validate_grade_record`
checks this schema; production projection also joins those hashes against the
frozen full schedule and raw evidence. Only the explicit projection produces
the compact `item_id`, `seed`, `status`, `correct`, optional `score` rows
accepted by `compare_paired`.

From the repository root, using a separate evaluator environment:

```bash
python3 -m venv .local/evaluator-env
.local/evaluator-env/bin/python -m pip install -e . -r configs/grading-requirements.txt
.local/evaluator-env/bin/python -m weight_space_lab acquire-ifeval --out .local/ifeval
.local/evaluator-env/bin/python -m weight_space_lab check-graders --ifeval-root .local/ifeval --out .local/known-answer-checks.json
.local/evaluator-env/bin/python -m unittest discover -s tests -p test_grading.py -v
```

Acquisition refuses existing outputs. It is bounded to 15 MB total, 6 MB per
file, a 300-second elapsed budget checked between reads, and 60-second socket
operations; a final socket operation may extend beyond the elapsed deadline.
Grading is entirely offline once dependencies and assets are available.

The known-answer report covers equivalent fractional and symbolic forms,
missing answers, multiple boxes, balanced and unbalanced malformed output,
JSON that is not a mathematical answer, incorrect and equivalent numeric
answers, strict/loose IFEval differences, instruction versus prompt metrics,
pinned sentence tokenization, evaluator failure, both truncation policies,
reasoning separation and pre-opened Qwen thinking. Missing optional evaluators
make checks fail; they are not silently skipped. Offline unit tests additionally
inject grader timeout, invalid process JSON, semantically invalid grader JSON,
model timeout and policy mismatch. None of these fixtures run inference.
