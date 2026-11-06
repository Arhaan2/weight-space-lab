# Implementation roadmap

## Phase 0: experimental mechanics — initial release

Implement inspectable, bounded patch operations; deterministic state hashes; configuration and split checks; paired result analysis; and synthetic invariant tests. Run a mechanics example without pretending its outputs are language-model scores.

**Exit:** tests and the example pass, the unfinished study configuration fails run-readiness checks, and public files contain no development-session material or machine-specific paths.

## Phase 1: experiment profiles and evidence contracts

Implement profile-specific configuration requirements for artifact acquisition, exact reconstruction, development smoke runs, fixed-patch evaluation, fresh-editor search, frozen final evaluation, and browser evaluation. Keep deterministic decoding explicit and define a separate item/sample/seed schedule for each benchmark. Preserve missing execution information as a blocker.

Provide a diagnostic command for dependencies, device availability, memory, and backend probe state without exporting machine paths or credentials. Define versioned raw run records with outputs, checkpoint and configuration identity, attempt history, and failure accounting; keep the compact analysis format stable. Report build, fixture-validation, real-model-validation, and experiment status independently.

**Exit:** CPU-only tests exercise these contracts and existing mechanics. An unfinished configuration cannot pass readiness. Synthetic validation cannot establish model execution or experimental success. This milestone does not include model acquisition, inference adapters, graders, editor access, or an experiment.

## Phase 2: pinned model acquisition and complete published-patch reconstruction

Acquire only the selected patch artifacts and matching base snapshot at the immutable revisions in `configs/published_p50c.json`. Record resolved source revisions, file hashes, stored tensor names/shapes/dtypes/counts, the actual shard index, model configuration, tokenizer assets, and chat template. Keep the original snapshot immutable and large weights outside tracked source.

Replay the sparse substitutions through the actual model-directory layout. Preserve unrelated tensors and assets, verify every published base and edited tensor hash, independently compare saved bytes, and report targeted column scope separately from actual BF16 changes. Finalize the model and audit together. This reconstructs a released intervention; it does not establish independent Astra discovery. See [the artifact workflow](model-artifacts.md).

**Exit:** fixture tests pass; the pinned real artifacts and saved reconstruction verify when downloads are available, or their exact prerequisites remain explicit. Artifact verification does not imply model loading, inference, grading, or an experiment. Native execution work follows this artifact boundary.

## Phase 3: native generation and resumable execution

Pin an architecture-compatible native backend and make tokenizer, template,
thinking, precision, sampling, stopping and limits explicit. Preserve the
released presence penalty without substituting a repetition penalty. Execute
a predetermined item/sample schedule with Phase 1 raw records, immutable
attempt history, bounded retries, hard worker timeouts, and validated resume.

Verify original, identity-copy and reconstructed checkpoints through the same
inference path using only a tiny development smoke. Record actual resources
and clearly distinguish short smoke settings from the released reference.
Test interruption, missing evidence, incompatible resume and unsupported
settings offline. Fixtures never produce scientific scores.

**Exit:** the implementation and recovery checks pass; tiny real generations
and resource measurements are recorded when the selected hardware/runtime is
available, otherwise their exact prerequisite remains explicit. No benchmark
experiment or editor search is launched. See [native execution](native-execution.md).

## Phase 4: benchmark data, independent graders, and paired analysis

Prepare pinned MATH, GSM8K, and IFEval sources and explicit development,
validation, and final-test manifests, retaining source identities. Audit exact
and near-duplicate or template-family overlap with dataset-specific policies.
Keep MATH-500 identities outside candidate-selection pools.

Use a pinned mathematical parser/verifier, a documented GSM8K answer policy,
and the official pinned IFEval evaluator. Validate them on independently
prepared known-answer cases. Freeze reasoning/output separation, truncation,
generation retry, and evaluator failure policies before a run. Keep raw
execution, diagnostic grades, and compact analysis records separate and test
the explicit projection. Production analysis requires every scheduled item
and sample, including mutually absent, failed, and timed-out cases.

Prepare an editor-visible development bundle and evaluator-only held-out
bundles. Their names and paths are an organizational convention, not access
control. Enforced isolation is a Phase 7 prerequisite.

**Exit:** offline implementation and meaningful checks pass; real evaluator
and data checks are reported separately with any unavailable prerequisites.
No pilot, candidate selection, editor call, or final benchmark experiment is
part of this phase. See [benchmark evaluation](benchmark-evaluation.md).

## Deferred execution gates

The previous roadmap grouped data preparation into a controlled-pilot phase.
The Phase 4 boundary above supersedes that grouping. The work below remains
pending and must not be inferred from passing data or grading fixtures.

### Controlled pilot and candidate freeze

Choose and verify the execution hardware; retain exact model, precision,
tokenizer, template, and decoding. Run a bounded development/validation
experiment with explicit call, sample, token, time, and cost limits. Preserve
all proposals, failed candidates, and validation feedback. Compare decoding
controls and freeze the selected checkpoint and evaluation configuration by
the preregistered rule. A no-improvement result is valid.

The actual Astra execution interface and model identity must be recorded.
No editor identifier is invented and no model is silently substituted. Start
each proposal from the immutable base and verify saved tensor changes and an
identity round trip. The editor must receive only approved data and tools.

### Phase 7: enforced editor/evaluator access boundary

Enforce separate identities, storage/tool permissions, and artifact access
for editor and final evaluator. Prove that final tasks, targets, generations,
audit details, and grading feedback cannot enter candidate selection. Validate
this boundary before making an isolation claim; neither JSON declarations nor
separate bundle directories prove it. Preserve the scientific role and actual
identity of Astra when it is used as the experimental editor.

### Held-out evaluation

After the candidate, configuration, split audit, policy, and access boundary
are frozen, execute matched predeclared arms on the full final item/sample
schedule. Audit grading failures and compute paired item-level uncertainty.
Publish positive, negative, or inconclusive evidence with full denominators;
shorter or usable output alone is not semantic correctness.

### Browser extension

Verify architecture support and identity conversion before any comparison.
Convert unchanged and edited checkpoints identically. Remeasure quality and
browser resources separately across independent sessions. Do not modify the
shared upstream benchmark repository as part of this work.

### Reproducible public release

Publish permitted scientific artifacts, immutable pins, full failure and
budget accounting, reproduction commands, and limitations. Verify authorship,
licenses, and metadata before publication. A second suitable machine must
reproduce the reported analysis; any execution restriction remains explicit.
