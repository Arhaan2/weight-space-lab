# Weight Space Lab

Controlled experiments in which a frontier model proposes direct edits to an open-weight language model, followed by independent evaluation.

Maintainer: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

## Current status

**Experimental infrastructure through Phase 2 artifact preparation. No language-model improvement has been measured.**

The implementation provides pinned model acquisition, complete file and tensor inventories, transactional reconstruction of the published sparse BF16 patch, independent saved-checkpoint verification, auditable array edits, experiment profiles, execution diagnostics, versioned raw evidence contracts, exact split-overlap checks, and paired analysis of externally graded results. Its mechanics example uses small synthetic arrays. It does not call a frontier model, implement a language-model inference server, or grade MATH answers.

Build and fixture validation, real artifact verification, model execution, and experiments are separate evidence categories. Model execution and experiments remain unexecuted. Passing a schema check establishes configuration completeness, not externally verified provenance, model compatibility, or a scientific result.

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

Raw run-record arrays are validated with `check-records`, benchmark schedules with `check-schedule`, and independent status records with `check-status`. These commands do not grade outputs or promote fixture evidence into real-model evidence. The compact `compare` input remains unchanged.

## Mechanics and analysis

- Applies declared matrix edits to a fresh copy of a pinned base state; no arbitrary code is executed from patch files.
- Rejects malformed operations, edits outside an allowlist, nonfinite values, and edits beyond the configured scalar budget.
- Reports hashes, changed scalar counts, and weight-difference sizes.
- Checks exact task-text overlap and duplicate identifiers across development, validation, and test manifests.
- Compares paired, independently graded item/seed records, treating failed or missing scheduled results as failures and resampling at the item level.

The supported exploratory algebraic operations are deliberately limited. They do not recreate the original discovery session or the full exploratory freedom of a Python kernel. The separate published-patch replay path uses the authors' released exact evaluated differences, which avoids estimating the original direction vector. Configuration checks do not establish that a remote checkpoint matches its claimed revision, that splits are semantically distinct, or that an editor was isolated from a test set.

## Study boundaries

Use Qwen3.5-0.8B for the native math study only after loading and validating the exact model revision and chat template. Keep a matched unchanged baseline and decoding controls. Choose edits using development data and a separate validation set; freeze the selected checkpoint before the final test.

For the browser extension, first verify that the selected runtime can load and execute the exact architecture. Convert unchanged and edited checkpoints with the same toolchain and quantization. Browser latency and throughput alone cannot establish improved task accuracy.

Large checkpoints, raw working data, local session transcripts, and development instructions are excluded from the source repository. Reproducible releases should include scientific task inputs where permitted, checkpoint identifiers, patch specifications, accepted edit rationale, evaluator versions, and independently scored evidence. Keep test data unavailable to the experimental editor until candidate selection is frozen.

## Attribution and licensing

The implementation here is original project code under the MIT license. Model weights, datasets, and any future imported components retain their own terms. The motivating research is by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. BrowserLLM was developed separately by Gufran Mirza and Arhaan Aggarwal; this repository does not claim authorship of its existing code.

See [docs/sources.md](docs/sources.md) for primary sources and [docs/publication.md](docs/publication.md) for the release checklist.
