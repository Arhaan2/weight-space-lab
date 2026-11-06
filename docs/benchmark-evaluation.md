# Benchmark evaluation boundary

Phase 4 implements benchmark preparation, independent grading, and projection
into paired analysis. It does not run a pilot, select a candidate, call an
experimental editor, or measure model improvement. The prior native smoke
outputs remain short, ungraded development evidence.

## Data and evaluation flow

1. Acquire the [pinned sources](benchmark-data.md) with bounded downloads.
   Retain exact source identities and source-file hashes. Prepare explicit
   development, validation, and final-test manifests and the overlap audit.
2. Keep the development bundle available to the editor and the held-out
   bundles evaluator-only. Final task text, targets, generations, and detailed
   overlap findings must remain outside the editor's candidate-selection
   context. **Bundle paths do not enforce this. Phase 7 must enforce and test
   the access boundary before an isolation claim.**
3. Freeze the [grading policy](independent-grading.md), full scheduled
   item/sample manifest, arm identities, and named metric before execution.
   Keep smoke, pilot, and full experiments separate. Generation policies must
   agree with the policy frozen for grading and analysis.
4. Preserve raw generations and all execution attempts. Grade independently;
   save extraction diagnostics, verifier identity, and errors separately.
5. Use the [explicit projection and paired comparison](paired-analysis.md).
   Every scheduled sample stays in the denominator, including cases absent
   from both arms. Missing output, a runtime failure, an incorrect answer, and
   a grader failure are different outcomes.

A usable output, completed generation, well-formed JSON object, or successfully
parsed expression does not establish correctness. An expression must pass the
independent verifier; an instruction must pass the official instruction
checker. Grader failure means correctness is unresolved, even when an
operational score conservatively retains that sample as zero.

## Dataset and metric scope

MATH-500 is the final mathematics set; selection data come from the pinned
MATH training source with identity/text/template exclusion. GSM8K retains its
original train/test source identities and adds a declared selection split.
IFEval's development/validation/final partition is a study-specific split of
the pinned official source. Its final subset is not the full official
541-prompt benchmark. Keep full-source replay and a custom held-out evaluation
as distinct protocols, and freeze the intended scope before experiments.

IFEval reports strict and loose scores at both prompt and instruction level.
Averaging per-prompt instruction fractions gives equal weight to prompts;
the official instruction-level rate instead weights every scheduled
instruction equally. Label these estimands separately and cluster paired
uncertainty by source prompt. Neither is a best-of-samples score.

## Reproduction and status

See [benchmark data](benchmark-data.md), [independent grading](independent-grading.md),
and [paired analysis](paired-analysis.md) for exact commands and schemas. Run
the default offline suite without inference or evaluator dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Optional data decoding and evaluator packages belong in a separate environment
from the pinned native inference environment. Missing evaluator packages,
source files, tokenizer resources, or incompatible versions must surface as
explicit grader failures; there is no string-match or model-judge fallback.
Use a new artifact destination for each preparation or report; existing
outputs are not overwritten.

The [Phase 4 check report](../reports/benchmark-grading/README.md) records the
checks actually executed, remaining prerequisites, and separate implementation,
fixture, real-model, and experiment status. No inference or paid infrastructure
is needed to reproduce the offline implementation checks.
