# Weight Space Lab

Controlled experiments in which a frontier model proposes direct edits to an open-weight language model, followed by independent evaluation.

Maintainer: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

## Current status

**Experimental infrastructure through Phase 7 offline editor integration and enforced local isolation. Live Astra integration and the measured full replay result remain pending; no language-model improvement has been established.**

The implementation provides pinned model acquisition, complete file and tensor inventories, transactional reconstruction of the published sparse BF16 patch, independent saved-checkpoint verification, native generation adapters, bounded schedule execution with crash-safe resume, auditable array edits, experiment profiles, execution diagnostics, versioned raw evidence contracts, pinned benchmark loaders and split manifests, exact and template-family overlap audits, independent benchmark graders, and scheduled paired analysis. Phase 5 joins these into a frozen replay workflow with separate development smoke, development pilot, and full-run commands. Its mechanics example uses small synthetic arrays. It does not call a frontier editor. Optional pinned evaluators grade MATH and IFEval; GSM8K uses an explicit numeric final-answer policy.

Build and fixture validation, real artifact verification, development model execution, and experiments are separate evidence categories. See [native execution](docs/native-execution.md) for the pinned backends, methodological variations, three-arm smoke procedure, and recovery contract. Experiments remain unexecuted. Passing a schema check establishes configuration completeness, not externally verified provenance, model compatibility, or a scientific result.

The [real development smoke](reports/native-smoke/README.md) generated 16 tokens each from the original, identity-copy and reconstructed checkpoints using the explicit Transformers CPU/BF16 variation. All three hit the short token cap; no answer was graded. The report retains the initial failed load, successful raw outputs, recovery checks, and measured resource requirements. The preferred vLLM GPU path is pinned but remains unexecuted.

The [Phase 4 verification report](reports/benchmark-grading/README.md) records pinned benchmark preparation, 25 independent known-answer checks with the real optional evaluators, and complete-schedule analysis fixtures. These are data and evaluator checks, not model accuracy measurements. Phase 7 implements and tests local isolation; a future live editor transport must establish the same boundary.

The [Phase 5 replay workflow](docs/published-patch-replay.md) compares unchanged and reconstructed p50c weights under identical published-reference settings, with a matched identity control and a separately labeled temperature-1 comparison. It freezes benchmark-specific schedules, execution identities, grading rules, and paired item-level analysis after development validation. Full schedules are MATH-500 × 4, GSM8K test × 2, and all 541 IFEval prompts × 2; full IFEval replay is distinct from the Phase 4 held-out subset. Reports retain denominators, failures, truncations, tokens, runtime, improved/regressed items, and discrepancies. The preferred GPU execution and measured full result remain pending. Installing the project or running its tests never starts the full benchmark.

The [Phase 5 evidence report](reports/published-patch-replay/README.md) records 255 passing tests, 15 completed short development generations, 25 passing grader checks, frozen pilot/full inputs, and an explicitly pending 28,600-generation full study. All smoke generations reached the 16-token cap; they do not estimate benchmark accuracy.

The [Phase 6 candidate adapter](docs/candidate-editing.md) applies bounded edits
to the actual Safetensors tensor names, dtypes, and shard layout. It preserves
unrelated tensors, buffers, architecture and tokenizer assets, logs bounded
weight inspections, and independently verifies saved stored-bit and numerical
changes before publishing a candidate. Every independent proposal starts from
the immutable original. Its development demonstration is explicitly a
hand-authored engineering edit, not an Astra-discovered intervention.
The [development evidence](reports/candidate-editing/README.md) records the
saved-change counts, inspection disclosures, identity control, and tiny native
execution checks without making an accuracy claim.

The [Phase 7 editor controller](docs/editor-controller.md) adds a strict
provider-independent file handoff, an OS-isolated non-scientific fixture,
development-only tools and probes, and a crash-safe reservation ledger.
Timeouts, cancellation, retries, interrupted requests, and budget exhaustion
are enforced in code. The [interface audit](docs/editor-interface-audit.md)
records the actually exposed `gpt-6-astra` selector and Codex CLI `0.163.0`,
with unavailable live identity/version/billing left null. No supported live
adapter yet meets the isolation and hard-limit requirements. No Astra editor
interaction or experiment has run.

The [live-readiness diagnosis](reports/editor-live-readiness/README.md) traces
the precise remaining gates and records behavioral checks of the installed CLI.
`editor-live-readiness` reports them without dispatch; `editor-prepare-connection`
saves a zero-tool acknowledgement plan. The tested CLI budget features do not
serialize a provider hard token cap. No live connection test was launched and
no exhausted offline budget was reset.

The explicit [authenticated session variant](docs/authenticated-session-editor.md)
reuses that machinery for one Astra proposal and one baseline/candidate
development comparison, without a separate API key or nested frontier call.
It records the subscription route's unavailable provider generation cap and
preserves the strict automated protocol. The local workflow passes synthetic
integration tests; the [session boundary audit](docs/session-boundary-audit.md)
records why the actual desktop session cannot yet be safely launched. The
separate development attempt is prepared and unspent; no genuine Astra proposal
or native development comparison has run in this variant.

An explicit [manual exploratory handoff](docs/manual-session-handoff.md) now
accepts an operator-transferred Astra/Ultra proposal using that same unspent
capacity. This path does not launch the editor and leaves isolation unverified;
it cannot qualify as a controlled research run. The private packet and separate
import/comparison commands are prepared; actual proposal evidence is pending.

The pinned p50c artifacts have been acquired and the complete saved reconstruction verified: **130,188 changed stored BF16 entries**, with a **131,072-coordinate published column footprint**. All 486 unrelated tensors and the remaining model assets are unchanged. See the [artifact verification report and inventories](reports/p50c/README.md); these checks establish byte reconstruction, not task quality or inference compatibility.

The intended study begins with replay of the [released evaluated edit](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit), then an informed fresh-editor experiment based on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), then a browser-deployment extension using [BrowserLLM](https://github.com/gufranmirza/browser-llm-bench). These are separate claims. The study design and remaining work are in [docs/study-design.md](docs/study-design.md) and [docs/roadmap.md](docs/roadmap.md).

## Run the mechanics example

Python 3.11 or later is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m weight_space_lab demo --out artifacts/mechanics-demo
python -m weight_space_lab check configs/qwen_math_design.json
```

The last command validates the proposed design. To check whether all required run configuration is present:

```bash
python -m weight_space_lab check configs/qwen_math_design.json --require-ready
```

That command **should fail** for the shipped template: model, runtime, dataset, grader, and editor-version pins have deliberately not been invented. Complete and verify them on the chosen inference machine before freezing an experiment.

## Profiles, diagnostics, and evidence

For the pinned model and published patch, follow [acquisition and reconstruction](docs/model-artifacts.md). The workflow keeps the original snapshot read-only, resolves tensors through the real shard index, preserves every unrelated tensor and asset, and finalizes the reconstructed model and audit together. Acquisition uses bounded public downloads; reconstruction and verification are offline and require no inference framework. Sparse substitutions, the published column footprint, and actual saved BF16 changes are recorded separately.

Version 2 protocols declare their execution purpose and require only the corresponding prerequisites. Published-patch reconstruction requires no editor connection. Fresh-editor search records the actual experimental editor's identity and interface, while final evaluation additionally requires a candidate and evaluation freeze. The original version 1 design remains supported without silently changing its settings.

See [experiment profiles](docs/protocol-profiles.md) for the requirement matrix, benchmark-specific schedules, and incomplete templates; [execution diagnostics](docs/diagnostics.md) for device and dependency checks; and [evidence records](docs/evidence-records.md) for raw outputs, retry accounting, independent grades, and status records.

```bash
python -m weight_space_lab diagnose
python -m weight_space_lab diagnose --probe-backends
python -m weight_space_lab check configs/profiles/native_development_smoke.json
python -m weight_space_lab check configs/profiles/native_development_smoke.json --require-ready
```

The final command is expected to exit with status 2 until its explicit missing requirements are supplied and verified. Diagnostics load no checkpoint and measure no inference performance. Optional backend probes are bounded; they neither install a runtime nor choose a model or precision. Default CI installs only the project's lightweight dependencies and runs CPU-only contract tests.

Raw run-record arrays are validated with `check-records`, benchmark schedules with `check-schedule`, and independent status records with `check-status`. These commands do not grade outputs or promote fixture evidence into real-model evidence. The compact `compare` input remains unchanged and is a low-level analysis utility. Production grading and comparison require a frozen complete item/sample manifest through the [Phase 4 pipeline](docs/benchmark-evaluation.md). Scientific raw-record projection rejects fixture evidence; its separate synthetic projection retains an explicit fixture envelope.

## Mechanics and analysis

- Applies declared matrix edits to a fresh copy of a pinned base state; no arbitrary code is executed from patch files.
- Rejects malformed operations, edits outside an allowlist, nonfinite values, and edits beyond the configured scalar budget.
- Reports hashes, changed scalar counts, and weight-difference sizes.
- Prepares explicit development, validation, and final-test manifests; checks exact identities, normalized text, numeric templates, and dataset-specific family overlap.
- Compares paired, independently graded item/seed records, treating failed or missing scheduled results as failures and resampling at the item level.

The supported exploratory algebraic operations are deliberately limited. They do not recreate the original discovery session or the full exploratory freedom of a Python kernel. The separate published-patch replay path uses the authors' released exact evaluated differences, which avoids estimating the original direction vector. Configuration checks do not establish that a remote checkpoint matches its claimed revision, that splits are semantically distinct, or that an editor was isolated from a test set.

## Study boundaries

Use Qwen3.5-0.8B for the native math study only after loading and validating the exact model revision and chat template. Keep a matched unchanged baseline and decoding controls. Choose edits using development data and a separate validation set; freeze the selected checkpoint before the final test.

For the browser extension, first verify that the selected runtime can load and execute the exact architecture. Convert unchanged and edited checkpoints with the same toolchain and quantization. Browser latency and throughput alone cannot establish improved task accuracy.

Large checkpoints, raw working data, local session transcripts, and development instructions are excluded from the source repository. Reproducible releases should include scientific task inputs where permitted, checkpoint identifiers, patch specifications, accepted edit rationale, evaluator versions, and independently scored evidence. Prepare separate editor-visible development and evaluator-only held-out bundles. These directory labels do not enforce access control; Phase 7 must enforce the editor/evaluator boundary before an isolated experimental claim.

## Attribution and licensing

The implementation here is original project code under the MIT license. Model weights, datasets, and any future imported components retain their own terms. The motivating research is by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. BrowserLLM was developed separately by Gufran Mirza and Arhaan Aggarwal; this repository does not claim authorship of its existing code.

See [docs/sources.md](docs/sources.md) for primary sources and [docs/publication.md](docs/publication.md) for the release checklist.
