# Research harness v2 — implementation validation

Validated on November 11, 2026. This record is engineering evidence, not a new
model accuracy or resource result. No Qwen forward, conversion or WebLLM browser
benchmark was executed for this harness update.

**105 targeted checks passed** across research checkpoint/deployment, persistent
kernel, diagnostic probes, orchestration, and existing mechanistic/joint-browser
regression suites. The run completed in 9.582 seconds.

Coverage includes arbitrary attention/MLP/norm/convolution edits, BF16 rounding,
fused-projection and quantization spillover audits, immutable snapshots/restore,
practice/final separation, failed-attempt accounting, final freeze closure and
one final browser run per arm. Fake model runtimes exercise token probabilities,
attention/activation hooks, paired generations, grading records and replay counts.

Actual macOS Seatbelt tests used tiny synthetic tensors: persistent Python state,
source/controller file denial, network/process denial, timeout/explicit-stop
cleanup, and daemon-death handling. A synthetic diagnostic-controller SIGKILL
test verified worker/descendant cleanup. These are not target-model calls.

The synthetic kernel environment was the existing local converter environment
(Torch 2.15.0), selected explicitly for the test. Source diagnostic inference
continues to require the separate pinned native-cpu runtime (Torch 2.10.0 /
Transformers 5.3.0). Actual model probes, conversion and final paired browser
execution remain unrun for this new workflow.

The [workflow](../../docs/research-browser-workflow.md) describes the tools,
budgets, setup commands, original-experiment correspondence and backend limits.
