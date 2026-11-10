# Implementation roadmap

The current executable workflow is the [bounded browser quality and resource study](ifeval-browser-quality-resource-protocol.md), with commands in the [manual runbook](ifeval-browser-v2-runbook.md). The phased plans below are historical context and do not impose additional prerequisites on that workflow.

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

## Phase 5: fixed published-patch replay

Connect the verified original, independently copied identity, and reconstructed
p50c checkpoints to the native runner, pinned benchmark data and graders, and
paired item-level analysis. Validate the complete pipeline on development
examples before freezing the evaluation. Keep smoke, pilot, and full execution
as separate explicit commands; installation and tests never launch the full
benchmark.

Freeze checkpoint and dataset identities, benchmark-specific item/sample
schedules, runtime, precision, tokenizer, chat template, prompt wrapping,
decoding and stopping, extraction/truncation/retry rules, graders, and primary
comparison/analysis settings. The primary comparison is unchanged versus
reconstructed p50c under identical settings. Include the identity control and
a separately labeled temperature-1 comparison with both checkpoints; the
alternative is fixed in advance and not selected from final feedback.

Use the published repeat counts: MATH-500 four samples per item, GSM8K two,
and full-source IFEval two. Full IFEval replay restores all 541 prompts and
does not constitute the Phase 4 custom held-out IFEval evaluation. Preserve
all declared runtime, dataset, random-stream, and grading discrepancies.

Reports retain average per-generation correctness, paired item-level
uncertainty, full denominators, missing/error/truncation counts, token use,
runtime, and improved/regressed items. Do not relabel average correctness as
best-of-N success. The scientific label remains **published-patch replay**
regardless of agreement with the reported improvement. Keep final outputs
outside the later fresh-editor workspace without claiming that directory
separation establishes Phase 7 access isolation.

**Exit:** the executable workflow and development checks are complete; use
measured smoke performance to estimate the full workload. The measured full
result remains explicitly pending while suitable compute is unavailable.
See [published-patch replay](published-patch-replay.md) for commands, frozen
scope, reporting, resource estimates, and discrepancies.

## Phase 6: real candidate-checkpoint editing

Adapt the bounded scale-column and direction-removal operations to complete
Safetensors checkpoints with their actual tensor names, dtypes, shard layouts,
and nonfloating buffers. Define binary64 arithmetic and per-operation rounding
back to storage dtype. Preserve architecture, tokenizer, unrelated tensors,
and all other assets. Avoid model-wide conversion or framework reserialization.

Bind each candidate to the immutable original; ordered operation composition
must be explicit. Enforce exact tensor/column/operation allowlists, finite
values, shape preservation, work limits, targeted and stored-change budgets,
and intervention parameter limits. Provide bounded metadata, statistics and
slice inspections with persistent exposure accounting.

Reopen saved files, verify unchanged tensors/assets and expected operation
results, and separately count stored-bit and numerical differences. Publish
model and audit together only after verification. Identity, valid-edit,
rejected-edit and failed-write checks must preserve the original and prevent
partial acceptance.

**Exit:** the implementation and fixture checks pass, and a deliberately
hand-authored engineering edit demonstrates the real-checkpoint path on
development input. Its provenance is never described as Astra discovery or
scientific improvement. See [candidate editing](candidate-editing.md) for the
operation contract, restrictions, inspection ledger, and development commands.

## Phase 7: editor interface, isolation, and probe controller

Inspect the supported execution interfaces and retain actual exposed model
identity/version and interface provenance without inventing an API identifier
or substituting another model. Implement a strict provider-independent
request/response contract and file handoff while live access remains pending.
Keep simulated and unauthenticated responses explicitly non-scientific.

Use an OS-enforced editor process boundary and a broker exposing only approved
model information, development examples, bounded weight inspections, patch
proposals, and frozen forward-pass probes. Deny host/evaluator files, network,
subprocesses, and inherited descriptors. A new chat is insufficient. No training,
gradients, optimizer operations, architecture changes, or evaluation edits are
available. Validate saved candidates through the Phase 6 adapter.

Reserve budget durably before dispatch; retain requests, retries, requested/
completed/failed sequences, tokens, elapsed time, and available billed usage.
Enforce cancellation and limits in supervised workers. Persist budget and
deadlines across restarts; reconcile interrupted requests conservatively.

**Exit:** offline integration, actual local denial checks, budget exhaustion,
recovery, and invalid-proposal verification are implemented. The exposed selector
is `gpt-6-astra`; the inspected CLI is `0.163.0`. A live isolated editor transport
with hard caps is still pending, so no real Astra interaction or experiment is
claimed. See [editor controller](editor-controller.md),
[interface provenance](editor-interface-audit.md), and
[verification evidence](../reports/editor-integration/README.md).

The [live-readiness follow-up](../reports/editor-live-readiness/README.md)
adds an explicit diagnostic gate and an inert zero-action connection plan.
Installed CLI behavior confirms that the tested token-budget features do not
send a provider hard output cap. Actual live launch/tool isolation and an
authorized live budget remain prerequisites. No Phase 8 search is started by
these diagnostics.

## Deferred execution gates

Fresh-editor experiments remain pending and must not be inferred from passing
fixtures, running the fixed replay, or demonstrating the engineering edit.

### Fresh-editor controlled pilot and candidate freeze

Choose and verify the execution hardware; retain exact model, precision,
tokenizer, template, and decoding. Run a bounded development/validation
experiment with explicit call, sample, token, time, and cost limits. Preserve
all proposals, failed candidates, and validation feedback. Compare decoding
controls and freeze the selected checkpoint and evaluation configuration by
the preregistered rule. A no-improvement result is valid.

Record the actual Astra execution interface and model identity without
inventing an editor identifier or silently substituting a model. Start each
proposal from the immutable base and verify saved tensor changes and an
identity round trip. The editor receives only approved data and tools through
the enforced access boundary.

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
