# Experiment profiles

Version 2 configurations make the execution purpose explicit. `validate_protocol` distinguishes a valid design from a complete run configuration. `check` validates the design; `check --require-ready` also returns every missing run requirement and exits with status 2 when any remain. Neither command verifies remote revisions, downloads artifacts, establishes editor isolation, or performs inference.

The original version 1 design and its interpretation remain supported. Migration is explicit: adding a profile does not change the legacy model, sampling parameters, benchmark, or numerical precision.

## Profile boundaries

| Profile | Purpose | Relevant prerequisites |
| --- | --- | --- |
| `artifact_acquisition` | Locate immutable model and released patch inputs | Model and source identity, immutable revisions, declared artifact references |
| `published_patch_reconstruction` | Reconstruct the released BF16 substitutions | Acquisition provenance and local checkpoint/manifest/archive hashes; the replay command checks expected tensor hashes |
| `native_development_smoke` | Bounded development-only generation | Model, tokenizer, chat template, runtime, precision, development tasks and explicit decoding |
| `fixed_patch_evaluation` | Evaluate a pre-existing published intervention | Native execution pins, fixed patch identity, benchmark-specific schedule and grader |
| `fresh_editor_search` | Propose candidates using approved development and validation feedback | Native execution pins, actual editor identity/version/interface, search limits, approved data and intervention policy |
| `frozen_final_evaluation` | Evaluate the selected intervention on final tasks | Matched inference, held-out schedule and grader, checkpoint/configuration/evaluator freeze, and actual editor provenance for an editor-derived intervention |
| `browser_evaluation` | Evaluate paired converted artifacts in browser sessions | Native checkpoint traceability, matched conversion/runtime/quantization, browser and session provenance, correctness tasks and grader; editor provenance when applicable |

The exact required fields and pending requirements are returned by the checker for each template in `configs/profiles`. Artifact preparation and published replay require no Astra connection or editor version. A fresh-editor final evaluation must retain the actual experimental editor provenance even when the editor is no longer connected during evaluation. Unknown model IDs, revisions, hashes, grader pins, runtime settings, and editor versions remain null until obtained and verified.

`stage` distinguishes `preparation`, `smoke`, `pilot`, and `full`; a successful development smoke run is not a pilot or a final experiment. Profiles constrain the allowed stages and splits. Full fixed-patch test evaluation also requires a freeze; browser test evaluation requires one even during a pilot. Final claims must match intervention origin: published patches use `published_patch_replay`, fresh editors use `informed_replication` or `extension`, and unchanged final controls use `baseline_evaluation`.

Frozen configurations bind the candidate, schedule, configuration, and evaluator. The configuration digest excludes the freeze object itself to avoid a self-referential hash; it includes the declared frozen state and all other configuration fields. Compute the configuration hash only after setting `frozen` to true. Changing covered settings invalidates that digest.

## Decoding and benchmark schedules

Each benchmark declares its own sampling configuration and seed list. No benchmark inherits an implicit sample count from another benchmark. Explicit `stochastic` mode requires a positive temperature; `greedy` uses temperature 0, top-p 1 and top-k 0. Thinking mode, presence penalty, and the generation limit remain explicit. Greedy decoding describes the algorithm, not bitwise reproducibility across runtimes or hardware.

The template's null benchmark settings are requirements, not runtime defaults. Populate them from the chosen protocol or verified source. Do not infer benchmark-specific settings from a general example or silently substitute a convenient model, dtype, task set, or generation cap.

`build_benchmark_schedule(benchmark, item_ids, seeds)` expands an explicit uniform seed schedule into rows with `benchmark`, `item_id`, `sample_index`, and `seed`. `validate_benchmark_schedule` rejects duplicate identities, inconsistent repeats or seeds within a benchmark, and malformed indices. Different benchmarks may use different schedules. A benchmark may have separate development and validation plans with disjoint item IDs and the same seed schedule. Its digest is independent of row ordering.

```python
from weight_space_lab.protocol import build_benchmark_schedule, validate_benchmark_schedule

# Synthetic contract illustration only; these are not benchmark settings.
schedule = build_benchmark_schedule("fixture", ["toy-a", "toy-b"], [7, 11])
summary = validate_benchmark_schedule(schedule)
```

Keep each benchmark's analysis separate. The compact analysis format still uses `item_id` and `seed`; richer execution records must be validated, filtered to one run and benchmark, and joined with independent grades before conversion.

## Local checks

```bash
python -m unittest discover -s tests -v
python -m weight_space_lab check configs/profiles/artifact_acquisition.json
python -m weight_space_lab check configs/profiles/native_development_smoke.json
python -m weight_space_lab check configs/profiles/frozen_final_evaluation.json --require-ready
```

The final command intentionally fails until the pending requirements are supplied. The source-only acquisition configuration can be complete without evidence that a download or reconstruction happened. The existing `replay` command still takes local inputs and checks published tensor hashes; profile checks do not replace that verification.

Phase 1 ends at these contracts and CPU-only fixtures. Real reconstruction requires the exact local artifacts referenced by `configs/published_p50c.json`. Generation additionally requires a compatible installed runtime, verified immutable tokenizer/model assets and explicit precision, followed by a bounded development smoke run. Benchmark evaluation requires task loaders and independently validated pinned graders. Fresh search additionally requires the actual Astra execution interface and enforced data isolation; browser evaluation requires the paired converted artifacts and a real browser session. Those implementations and executions belong to subsequent milestones.
