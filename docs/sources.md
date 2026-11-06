# Primary sources

Reviewed 2026-11-06. Pin source and dependency revisions when implementing the real experiment; a mutable URL is background context, not a reproducibility pin.

1. Takehi, Teoh, Zhang, and Khattab, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Motivating experimental report. Its [Experiment 8 release](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md) includes exact evaluated edits. This project has [verified the saved byte reconstruction](../reports/p50c/README.md), but has not run model inference or reproduced the reported accuracy.
2. Qwen, [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B). Official architecture, model usage, licensing, and inference recommendations. The documented thinking-mode text recommendation uses temperature 1.0, motivating a control alongside the study's temperature 0.6.
3. Gufran Mirza and contributors, [BrowserLLM source](https://github.com/gufranmirza/browser-llm-bench), and [public site](https://browserllm.dev/). Existing browser measurement project; assess its actual current capabilities before integration.
4. Hugging Face, [Math-Verify](https://github.com/huggingface/Math-Verify). Candidate mathematical answer verifier. Its configuration and revision must be validated and pinned; it is not implemented in this initial release.
5. HuggingFaceH4, [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), with [original split provenance](https://github.com/openai/prm800k). Candidate evaluation data; keep separate from development pools.
6. Google Research, [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval). Official instruction-following evaluation code. Preserve the distinction between strict/loose and prompt/instruction-level metrics.

No external project code or model weights are bundled in this release.

## Exact replay references

`configs/published_p50c.json` pins the upstream source revision and checkpoint revision. The upstream [manifest](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json) supplies expected base and edited tensor hashes. The [patch archive](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/edit.npz) contains the exact sparse BF16 substitutions. Reference these artifacts directly; do not copy the authors' complete repository history or conversation logs into this project.
