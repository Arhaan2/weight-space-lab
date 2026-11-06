# Example commands

Run the synthetic mechanics example from the repository root:

```bash
python -m weight_space_lab demo --out artifacts/mechanics-demo
```

It produces a patch proposal and audit, an explicit item/seed schedule, two arrays of synthetic grades, and a clearly labeled mechanics report. No language model is loaded. The two synthetic arms have equal aggregate grades deliberately; the example makes no capability-improvement claim.

Reanalyze those synthetic records:

```bash
python -m weight_space_lab compare \
  --schedule artifacts/mechanics-demo/schedule.json \
  --baseline artifacts/mechanics-demo/baseline.json \
  --candidate artifacts/mechanics-demo/candidate.json \
  --out artifacts/mechanics-demo/reanalysis.json
```

For real records, retain the same `item_id`, `seed`, `status`, and boolean `correct` schema. Allowed statuses are `ok`, `truncated`, `error`, `timeout`, and `missing`. A scheduled record absent from an arm is counted as a missing failure. Never use the synthetic examples as benchmark data. The analysis does not validate the truth of an upstream correctness label.

Accuracy can retain a correctly graded truncated answer with `--truncation-policy grade`, or reject every truncated answer with `--truncation-policy fail`. Choose the policy before evaluation. The `completion_within_budget` mode counts only correct records classified as `ok`; an upstream runner must enforce the actual resource and timing budgets. This utility does not measure those budgets itself.

## Replay a real published sparse edit

The inputs below are local files that must first be obtained from the immutable revisions in `configs/published_p50c.json`. Download only the model and edit artifacts required for replay, preserving their source attribution. Keep weights in the ignored `checkpoints` directory and upstream patch files in ignored `artifacts`.

```bash
python -m weight_space_lab replay \
  --base checkpoints/original/model.safetensors \
  --manifest artifacts/upstream-p50c/manifest.json \
  --archive artifacts/upstream-p50c/edit.npz \
  --out checkpoints/replayed/model.safetensors \
  --audit artifacts/p50c-replay-audit.json
```

The replay utility supports the sparse BF16 format used for p50c, not the different low-rank format used for other runs. It preserves the base checkpoint and refuses to overwrite an existing destination. Hash agreement verifies tensor reconstruction, not benchmark accuracy or independent edit discovery. Copy the matching tokenizer/model configuration separately before loading the result into an inference runtime.
