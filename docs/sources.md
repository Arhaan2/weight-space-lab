# Primary sources

Reviewed 2026-11-06. Pin source and dependency revisions when implementing the real experiment; a mutable URL is background context, not a reproducibility pin.

1. Takehi, Teoh, Zhang, and Khattab, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Motivating experimental report. Its [Experiment 8 release](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md) includes exact evaluated edits. This project has [verified the saved byte reconstruction](../reports/p50c/README.md) and [tiny native generation](../reports/native-smoke/README.md), but has not reproduced the reported accuracy.
2. Qwen, [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B). Official architecture, model usage, licensing, and inference recommendations. The documented thinking-mode text recommendation uses temperature 1.0, motivating a control alongside the study's temperature 0.6.
3. Gufran Mirza and contributors, [BrowserLLM source](https://github.com/gufranmirza/browser-llm-bench), and [public site](https://browserllm.dev/). Existing browser measurement project; assess its actual current capabilities before integration.
4. Hugging Face, [Math-Verify](https://github.com/huggingface/Math-Verify). The optional Phase 4 mathematical verifier is pinned to Math-Verify 0.9.0 and its exact parser dependencies; see [independent grading](independent-grading.md). Its known-answer validation is fixture evidence, not model accuracy.
5. HuggingFaceH4, [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), with [original split provenance](https://github.com/openai/prm800k). Final mathematics data pinned in [benchmark sources](../configs/benchmark_sources.json); its 500 source identities remain separate from selection pools. The training mirror retains shard/row identities because it does not expose original MATH filenames.
6. Google Research, [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval). Official instruction-following evaluation code. Preserve the distinction between strict/loose and prompt/instruction-level metrics.

No external project code, benchmark task contents, or model weights are bundled in tracked source. Explicit acquisition keeps official evaluator code and its upstream license together in local artifacts. Dataset terms and upstream authorship remain in effect.

## Exact replay references

`configs/published_p50c.json` pins the upstream source revision and checkpoint revision. The upstream [manifest](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json) supplies expected base and edited tensor hashes. The [patch archive](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/edit.npz) contains the exact sparse BF16 substitutions. Reference these artifacts directly; do not copy the authors' complete repository history or conversation logs into this project.

## Phase 4 data and evaluator pins

The [source manifest](../configs/benchmark_sources.json) records verified
immutable revisions, original source locations, file sizes, row counts and
SHA256 digests for all eleven benchmark files. It uses the
[EleutherAI MATH training mirror](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3),
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be),
[original GSM8K](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c),
and [official IFEval](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval).

The official IFEval adapter calls the unchanged strict and loose evaluator
functions, preserving Google Research's Apache-2.0 source license. Its
sentence-tokenizer data are pinned separately to
[NLTK data revision 4f15a3d](https://github.com/nltk/nltk_data/tree/4f15a3d89eefe9748ec1c05be495d91289197155).
See [benchmark preparation](benchmark-data.md) for the documented MATH source
access limitation, custom IFEval split, quarantine rules, and overlap-audit
limitations. These choices are explicit study preparation, not a claim to
reproduce a published benchmark score.
