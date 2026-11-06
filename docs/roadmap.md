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

## Phase 3: native execution and bounded edit execution

Choose an execution machine and verify available memory and inference performance with a bounded development-only smoke run. Pin runtime dependencies and exact inference settings without changing the acquired model, precision, tokenizer, or chat template silently.

Add the native generation adapter, task loaders, pinned graders, and machine-readable item/seed records. Verify one development task end to end. Test the model-card baseline and literature baseline on development data. Freeze final test identities without showing their contents to the experimental editor. Treat benchmark evaluation of the fixed published patch as its own replay study; it is not a source of test feedback for the later fresh-editor study.

Implement Safetensors loading/saving with support for the selected model's actual dtypes and shard layout. Verify an identity round trip before applying any candidate. Preserve all nonedited tensors, tokenizer assets, model configuration, and architecture. Prove the actual changed-coordinate count on the saved/reloaded checkpoint.

Connect the limited proposal format to an explicitly recorded Astra execution interface. Do not invent an API model identifier or quietly substitute another model. Add an execution-side probe ledger that enforces call, sequence, token, time, and cost limits before work begins. Run the editor in a context that cannot access final-test files.

**Exit:** original and replayed checkpoints generate real outputs, graders pass independent examples, and every scheduled result is accounted for. One real bounded edit is loadable and auditable; prohibited operations cannot pass the execution interface; a failed candidate restores the pristine base. No fresh candidate selection uses final-test feedback.

## Phase 4: controlled pilot and candidate freeze

Run a bounded development/validation experiment. Preserve all candidate proposals, weight differences, probe budgets, and validation failures. Compare to decoding-only controls. Select according to the preregistered rule and freeze checkpoint hash, configuration hash, and evaluation code.

**Exit:** a selected candidate or an explicit no-improvement result, with enough evidence to reproduce the search procedure. No selection on the final test.

## Phase 5: held-out evaluation

Execute the same final item/seed schedule for all predeclared arms. Grade independently, audit extraction failures, calculate paired item-level intervals, and publish full denominators and failure accounting. Add transfer and regression checks before broadening the claim.

**Exit:** an evidence-backed positive, negative, or inconclusive conclusion. A shorter generation is reported as a shorter generation unless correctness also improves.

## Phase 6: BrowserLLM extension

Pin and audit BrowserLLM. Verify architecture support, add custom artifact references if needed, and add or validate task-correctness grading. Convert base and edited models identically, remeasure quality, then run resource comparisons across independent browser sessions.

**Exit:** browser results separate quality, completion, latency, throughput, memory, and measurement limitations; base and edited artifacts are traceable to their native checkpoints.

## Phase 7: reproducible public release

Publish executable commands, immutable dependency pins, permitted data references, patches, checkpoint hashes, per-example evidence, all candidate budgets, analysis code, and limitations. Keep the project on the maintainer's account and verify attribution before each push.

**Exit:** a clean checkout on a second suitable machine reproduces the reported analysis and the documented evaluation path. Any remaining restriction is explicitly stated.
