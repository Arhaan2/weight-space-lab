# Phase 6 candidate-checkpoint editing

This is a **hand-authored engineering edit**, not an Astra-discovered
intervention. No editor was called, no candidate was selected using feedback,
and no language-model improvement is claimed.

The [report](report.json) records a fixed scale of column zero of
`model.language_model.layers.15.mlp.down_proj.weight` by 0.99 on the acquired
Qwen3.5-0.8B checkpoint. The original contains **488 tensors and 873,438,784
stored coordinates**, with BF16 and F32 storage. The exact architecture,
tokenizer, chat template, shard index, headers, and unrelated assets were
preserved.

- The independent identity candidate has **zero stored-bit and numerical
  differences**; all 488 tensors match the original.
- The engineering candidate has **1,024 stored-bit and 1,024 numerical
  differences**, within its 1,024-coordinate declared footprint. All 487
  unrelated tensors match the original. No signed-zero-only changes occurred.
- The maximum absolute stored change is 0.000244140625; the changed fraction
  is approximately 0.00000117238 of all stored coordinates.
- Both candidates were saved, reopened, and checked against the original,
  including independent operation-result recomputation. The original's file
  inventory remained unchanged.
- An out-of-policy factor of 0.5 was rejected without a candidate output.
  Injected failed-write, corruption, budget, dtype, composition, and exposure
  checks run in the offline suite, not against the real original.

The full offline suite passed **298 tests** using
`python -m unittest discover -s tests`. The tests cover identity, accepted
and rejected edits, failed writes, preserved buffers/assets, saved-byte
verification, rounding, signed zero, and bounded exposure accounting.

The [identity generation](identity-generation.json) and
[engineering generation](engineering-generation.json) each contain four
generated tokens from the same authored development input,
`Compute 7 + 5.` Both completed generation and reached the four-token cap;
their statuses are `truncated`. They are execution smoke checks, not completed
answers or accuracy measurements. The explicit backend is Transformers CPU
with the pinned BF16/F32-preserving implementation, greedy decoding, seed 17,
one request per arm, no retry, a 120-second startup limit, and a 60-second
generation limit. No final-test or validation input was read.

Three [logged inspections](exposures.jsonl) exposed only the declared target's
metadata, statistics for its first four rows in column zero, and those four
values. The [policy](policy.json), [proposal](engineering-proposal.json), and
[development input](development-input.json) are included. Public generation
records retain full output and native metadata; the full environment package
inventory is represented by its hash. Complete model bundles, full audits,
and unabridged runtime evidence remain in the ignored local artifact output.

A preliminary development run also generated four tokens per arm. Its report
initially classified `truncated` as no inference and returned a nonzero CLI
status. The classifier and regression test were corrected, and the complete
demo was rerun from the original into a fresh output. The preliminary bundle
is retained, and its hashes and execution accounting are recorded in the
report. Neither run informed the fixed engineering proposal.

Reproduce with the pinned native CPU environment and acquired original:

```bash
python -m weight_space_lab demo-candidate-edit \
  --model artifacts/p50c-acquisition/model \
  --out artifacts/phase06-engineering-native-v2 --backend transformers
```

Choose a new output path for another run; existing evidence is never
overwritten. See the [operation and inspection contract](../../docs/candidate-editing.md)
for restrictions relative to an exploratory Python kernel. Editor/evaluator
process isolation and a real Astra experiment remain separate future work.
