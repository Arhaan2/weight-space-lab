# BrowserLLM integration audit

Audited 2026-11-06 at [source revision b39187b](https://github.com/gufranmirza/browser-llm-bench/tree/b39187b1aff1f218493622d7fcf3cd5d24773b41).

## Existing capability

BrowserLLM is the browser performance project co-developed by Gufran Mirza and Arhaan Aggarwal. The [public site](https://browserllm.dev/) identifies its measurements as generation speed, latency, response time, memory, and resource use; it explicitly excludes answer accuracy.

The pinned [package configuration](https://github.com/gufranmirza/browser-llm-bench/blob/b39187b1aff1f218493622d7fcf3cd5d24773b41/package.json) includes WebLLM 0.2.82 and a separate 0.2.86 package alias. Its [catalog](https://github.com/gufranmirza/browser-llm-bench/blob/b39187b1aff1f218493622d7fcf3cd5d24773b41/catalog/models.json) already contains Qwen3.5-0.8B in q4f16_1, q4f32_1, and q0f16 forms using 0.2.86. The [publication manifest](https://github.com/gufranmirza/browser-llm-bench/blob/b39187b1aff1f218493622d7fcf3cd5d24773b41/public/reports/manifest.json) includes a Qwen3.5-0.8B-q4f16_1 report. Architecture support is therefore an existing foundation, not a hypothetical future feature.

## Changes needed for this study

The current [workloads](https://github.com/gufranmirza/browser-llm-bench/blob/b39187b1aff1f218493622d7fcf3cd5d24773b41/src/benchmark/protocol/workloads.ts) are open-ended explanation and summarization tasks without reference answers or semantic graders. Output validity must not be counted as accuracy.

The [settings schema](https://github.com/gufranmirza/browser-llm-bench/blob/b39187b1aff1f218493622d7fcf3cd5d24773b41/src/benchmark/protocol/settings.ts) limits output to 2,048 tokens and context to 4,096. Its normal defaults are a 64-token output cap, temperature 0, and thinking disabled. These are materially different from the native math study. A long-reasoning browser study needs a separately versioned protocol; a shorter practical extraction task can use a separately defined budget.

The existing worker loads catalog model URLs and compiled libraries. Convert unchanged and edited checkpoints identically, give the edited artifact its own immutable identity, and keep runtime versions matched. A native Safetensors file is not itself an MLC browser artifact. First compare native weights; then q0f16 pairs where feasible; then q4f16_1 pairs to quantify what quantization changes.

Add a versioned correctness task and grader, freeze a development/validation/test split, and preserve per-case outputs and settings. Independent browser sessions supply a separate source of variation from repeated requests in one session. Do not merge timing repetitions with the number of independent accuracy examples.

Reference upstream raw reports by URL. Do not copy them wholesale: their system-process metadata contains machine-specific paths that do not belong in this project's public evidence.
