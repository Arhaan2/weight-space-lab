# Weight Space Lab

Controlled experiments in which a frontier model proposes direct edits to an open-weight language model, followed by independent evaluation.

Maintainer: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

## Current status

**Experimental infrastructure, version 0.1. No language-model improvement has been measured.**

The initial release implements auditable array edits, experiment-configuration checks, exact split-overlap checks, paired analysis of externally graded results, and a small replay utility for the published sparse BF16 edit format. Its executable example uses small synthetic arrays to test the mechanics. It does not download a language model, call a frontier model, implement a language-model inference server, or grade MATH answers.

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

## What the first release does

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
