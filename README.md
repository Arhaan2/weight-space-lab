# Weight Space Lab

Controlled experiments in which a frontier model proposes direct edits to an open-weight language model, followed by independent evaluation.

Maintainer: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

## Next experiment — open weight research, one final comparison

The recommended [research harness](docs/research-browser-workflow.md) gives Astra
persistent CPU Python, arbitrary internal tensor edits, named candidate
snapshots, and original/edited/paired diagnostic generations and teacher-forced
probes (log probabilities, hidden states, MLP activations and full attention).
It permits dose, layer, neuron and random-control experiments without prescribing
a direction estimator or fixed edit family. Development has separate budgets
of up to 300 Python calls and 50 forward attempts.

One final snapshot closes editing before the **single baseline/candidate WebLLM
comparison**, measuring IFEval accuracy and resource usage on the same responses.
Practice and final prompts are separate. This is an implementation update; no new
model-performance result is claimed. Completed results below retain their
original methods and limitations.

## Current result — one mechanistic IFEval / browser pair

**The first directional edit did not improve accuracy and resource usage together.** Strict compliance fell from 16/32 to 15/32, with 1.25% more output tokens. Observed response time was 2.97% shorter, but time per compliant answer increased 3.50%. This is one exposed development comparison, not evidence of a robust speedup.

Astra inspected actual weights and source-model activations, then edited 127 noncontiguous columns in layers 19 and 22 with strength 2. Both models were graded and resource-measured on the same 32 IFEval responses. An initial candidate browser closure made zero model calls; its preserved evidence and explicit pre-load continuation are disclosed.

See the [complete result and limitations](reports/mechanistic-browser-v1/README.md), [canonical measurements](reports/mechanistic-browser-v1/joint-comparison.json), [exact intervention](reports/mechanistic-browser-v1/intervention.json), and [workflow instructions](docs/mechanistic-browser-workflow.md). IFEval is the only accuracy benchmark in this version. CPU diagnostic activations remain distinct from quantized WebGPU execution.

## Earlier results — fresh baseline and Astra round 4

**The new Astra edit did not improve benchmark compliance or demonstrate lower resource use.** All six artifacts were evaluated afresh on the same current browser: pristine, V1 and all four Astra edits. Round 4 scored 16/32 strict passes, equal to pristine and below V1's 17/32. It used 16,998 scored quality-output tokens versus 13,941 for pristine (+21.93%), with 16 versus 12 truncated outputs. Three prompt outcomes improved and three regressed; 26 were tied.

| Fresh artifact | Strict passes | Loose passes | Quality output tokens | Resource response (ms/request) | Sampled native peak (GB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pristine | 16/32 | 17/32 | 13,941 | 612.88 | 2.990 |
| V1 edit | 17/32 | 18/32 | 13,903 | 663.82 | 2.884 |
| Astra round 1 | 16/32 | 17/32 | 15,019 | 674.79 | 2.869 |
| Astra round 2 | 16/32 | 17/32 | 12,644 | 656.89 | 2.886 |
| Astra round 3 | 16/32 | 17/32 | 12,591 | 768.90 | 2.963 |
| Astra round 4 | 16/32 | 17/32 | 16,998 | 737.83 | 3.062 |

Resource columns are equal-weight means of two independent browser sessions per artifact, each with 60 measured fixed-prompt requests. Round 4's mean response was 737.83 ms versus 612.88 ms for pristine (+20.39%), and its mean sampled native peak was 3.062 versus 2.990 GB (+2.41%). The resource workload generated 48 tokens per request for round 4 versus 39 for the other artifacts (+23.08%); full-response throughput was 65.06 versus 63.64 tokens/s (+2.23%). Thus answer length, response cost and per-token speed must be interpreted together. GPUBuffer descriptor and WASM address-space maxima were unchanged allocation bookkeeping. Power was unavailable, and CPU/GPU activity rates are not cumulative computation or energy. Two sessions on one device do not establish significance, causality or resource equivalence.

The first five fresh quality results were supplied to one fresh-context collaboration agent requested with `gpt-6-astra`. It proposed scaling columns 0–59 of the layer 15 and 19 MLP down-projection matrices by 0.5, starting from pristine. Its 140-word prediction was saved before building or evaluating the candidate. Independent source and effective deployment audits passed. Provider immutable identity, accepted reasoning effort, tokens, billing and technical isolation remain unverified or unavailable. This was one proposal, not an unreported optimization search.

All 18 model sessions and 930 planned generation dispatches completed without a target failure or retry: 192 scored quality requests, 720 measured resource requests and 18 warm-ups. Both quality and resource cohorts match the recorded browser/device/runtime identities. The mirrored resource order, fixed workload, observed variation, finite limits, preparation failure and narrow post-session cache maintenance are retained. These are exposed development tasks, with **no untouched final-quality confirmation**.

The [interactive research website](https://arhaan2.github.io/weight-space-lab/) includes all six versions, an interactive quality/resource figure, individual sessions and requests, exact edits, on-page methodology, primary references and downloads. The [canonical fresh comparison](reports/fresh-baseline-r4-v1/result.json), [maintenance evidence](reports/fresh-baseline-r4-v1/maintenance.json), [frozen protocol](docs/fresh-baseline-r4-protocol.md) and [research methodology](docs/research-methodology.md) preserve the detailed record. Historical studies below remain separate.

## Earlier development search and resource pilot — November 10, 2026

**The combined goal is not established.** Three fresh Astra weight edits were independently audited, converted and evaluated in WebLLM on the same 32 development prompts. All three matched the original model's 16/32 strict passes. None met the frozen requirement of higher strict compliance with no more scored output tokens, so pristine was retained and the conditional Browser LLM Bench A–B–B–A comparison was skipped.

| Development artifact | Strict passes | Loose passes | Scored output tokens |
| --- | ---: | ---: | ---: |
| Saved original baseline | 16/32 | 17/32 | 13,941 |
| Astra round 1 | 16/32 | 17/32 | 15,019 |
| Astra round 2 | 16/32 | 17/32 | 12,644 |
| Astra round 3 | 16/32 | 17/32 | 12,591 |

Explore the [results website](https://arhaan2.github.io/weight-space-lab/), [completed execution report](reports/ifeval-browser-v2-collaboration-run/README.md) and [aggregate data](reports/ifeval-browser-v2-collaboration-run/result.json). The report preserves all failed stages, exact-file recovery and actual dispatch accounting. No additional target generation was used to recover offline grading.

The editor route used actual fresh-context `gpt-6-astra` collaboration agents; Ultra effort and immutable provider identity were not verified. The saved baseline used Chrome 156.0.8078.80 and the candidates used Chrome 157.0.8103.39. There are zero verified untouched final prompts, so this is development evidence with a browser-version confound, not a fresh held-out quality claim. Fewer output tokens alone do not establish lower browser memory, power or resource use.

The [frozen protocol](docs/ifeval-browser-quality-resource-protocol.md), [historical preparation report](reports/ifeval-browser-v2-narrowed/README.md) and [execution amendments](docs/2026-11-10-direction.md) explain the bounds and deviations. A subsequent quality claim needs a new baseline on the same current browser and a verified untouched final set.

A separate, subsequently authorized **all-edit resource pilot is complete**: pristine, V1, and all three Astra edits each ran two 60-request sessions in a mirrored schedule on the same recorded device, browser and runtime. Including one warm-up per session, this used 10 loads and 610 requests, with no target failures or retries. Every measured session generated 2,340 output tokens.

| Resource artifact | Mean response time (ms/request) | Mean sampled native peak (GB) |
| --- | ---: | ---: |
| Pristine | 617.85 | 3.092 |
| V1 edit | 601.06 | 3.075 |
| Astra round 1 | 652.13 | 2.995 |
| Astra round 2 | 601.70 | 3.054 |
| Astra round 3 | 1,055.28 | 2.895 |

These are equal-weight means of two sessions per artifact. GPUBuffer descriptor and WASM address-space maxima were identical across the schedule; they are allocation bookkeeping, not physical VRAM. Power was unavailable. Lower CPU/GPU activity rates do not establish less total work; round 3's lower native footprint coincided with slower responses, including both its 661.28 and 1,449.28 ms/request sessions.

An integrity preflight stopped after six completed sessions. The four unused positions ran through a documented continuation after a **22.20-minute inter-session gap**; no completed observation was replaced. Hardware/runtime matching does not imply uninterrupted or constant host conditions. This two-session pilot is descriptive and establishes neither resource equivalence nor causal efficiency gains. It does not change the negative V2 quality-selection result.

The [interactive research website](https://arhaan2.github.io/weight-space-lab/) exposes artifact comparisons, individual sessions and request traces, exact weight-edit maps, methodology, downloadable data and citation metadata. See the [resource report](reports/resource-all-edits-v1/README.md), [resource JSON](reports/resource-all-edits-v1/result.json), and [research methodology](docs/research-methodology.md) for measurement definitions, frozen source identities, recovery records and limitations.

## Closed results and historical preparation

These records retain prior milestones. Statements about pending execution or authorization below refer to their named historical workflows; the fresh six-version result is above.

**IFEval browser V1 is closed: completed experiment, inconclusive held-out compliance.** Strict prompt compliance was 70/128 candidate versus 69/128 baseline (+0.78125 percentage points; frozen 95% interval −5.47 to +7.03). Improvement and equivalence are unestablished. The candidate completed through a separate continuation after the preserved supervisor interruption. See the [canonical V1 results](reports/ifeval-browser-v1-final/README.md), [saved-output interpretation](docs/ifeval-v1-output-review.md), [public reproduction record](docs/ifeval-v1-reproduction.md), and [historical V2 proposal](docs/ifeval-browser-v2-protocol.md). That proposal is superseded by the current workflow above; its preparation evidence is retained unchanged.

**Phase 7 includes one completed manual exploratory proposal/checkpoint/development comparison, with editor isolation unverified. Strict automated Astra integration and the measured full replay result remain pending; no language-model improvement has been established.**

The implementation provides pinned model acquisition, complete file and tensor inventories, transactional reconstruction of the published sparse BF16 patch, independent saved-checkpoint verification, native generation adapters, bounded schedule execution with crash-safe resume, auditable array edits, experiment profiles, execution diagnostics, versioned raw evidence contracts, pinned benchmark loaders and split manifests, exact and template-family overlap audits, independent benchmark graders, and scheduled paired analysis. Phase 5 joins these into a frozen replay workflow with separate development smoke, development pilot, and full-run commands. Its mechanics example uses small synthetic arrays. It does not call a frontier editor. Optional pinned evaluators grade MATH and IFEval; GSM8K uses an explicit numeric final-answer policy.

Build and fixture validation, real artifact verification, development model execution, and experiments are separate evidence categories. See [native execution](docs/native-execution.md) for the pinned backends, methodological variations, three-arm smoke procedure, and recovery contract. Controlled research experiments remain unexecuted. Passing a schema check establishes configuration completeness, not externally verified provenance, model compatibility, or a scientific result.

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
adapter yet meets the isolation and hard-limit requirements. No runner-launched
Astra interaction or controlled research experiment has run.

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
records why the actual desktop session cannot yet be safely launched under the
strict protocol. The separate development capacity was used once through its
explicit manual exploratory extension below.

An explicit [manual exploratory handoff](docs/manual-session-handoff.md) now
accepted the externally supplied proposal attributed by the operator to
Astra/Ultra. The [sanitized result](reports/editor-session/manual-result.json)
records a validated checkpoint with 32,768 changed stored entries and one real
baseline/candidate development pair. Each arm generated three tokens and stopped
at EOS; no grader ran. This path does not launch the editor, leaves isolation
unverified, and cannot qualify as a controlled research run. The single proposal
and comparison capacity is spent; no retry or additional evaluation was run.

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
