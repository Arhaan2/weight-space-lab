# Study design

## Research question

Can GPT-6 Astra propose small direct weight edits that improve a fixed open-weight model on previously withheld tasks, and does any improvement survive deployment in a browser?

Separate the two claims. The native experiment measures task quality under controlled inference. The browser experiment measures whether the same change survives conversion, quantization, and a different execution environment. A successful first experiment does not establish the second.

This is an **informed replication**: the published mechanism has already been read. Do not describe it as a blind rediscovery. A genuinely blinded study would need a fresh editor context and a different, predeclared access policy.

## Track A0: exact replay

The authors released exact sparse substitutions for their evaluated p50c checkpoint. Use the source and base revisions in `configs/published_p50c.json`. Verify source tensors and reconstructed tensors against the published hashes before running inference. Count both targeted coordinates and actual changed stored values; BF16 rounding makes those different quantities.

Evaluate this fixed intervention with the original evaluation configuration, then the predeclared decoding controls. Label the resulting evidence patch replay. It does not demonstrate that a fresh editor found the intervention. Keep final test feedback from this replay outside the context used for a fresh discovery study; use a fresh held-out task set for extension claims if the original test has influenced design choices.

## Track A1: fresh editor and native accuracy

Start with the checkpoint and settings recorded in `configs/qwen_math_design.json`. It is a proposed configuration, not an executed result. Pin the complete checkpoint, tokenizer, chat template, dataset revisions, evaluator code, runtime, numerical precision, and the exact editor model version exposed by the execution interface.

Use separate development, validation, and final-test data. Do not use the final test to select layers, choose edits, tune decoding, change the scoring script, or decide when to stop. Exclude MATH-500 task identities from every pool used for candidate selection. Check normalized text as well as identities, then inspect near-duplicates and shared templates separately.

The experimental editor should receive weights and approved development feedback. The final evaluator should operate separately and reveal final scores only after the selected candidate and study configuration are frozen. A JSON declaration does not enforce this separation; the eventual execution system must limit the editor's filesystem, tools, and accessible artifacts.

### Comparisons

| Arm | Weights | Decoding | Purpose |
| --- | --- | --- | --- |
| Literature baseline | Original | Published configuration | Establish a reference condition |
| Model-card baseline | Original | Current vendor recommendation, pinned | Test sensitivity to default settings |
| Tuned baseline | Original | Chosen on development data | Establish whether decoding can solve the same problem |
| Identity control | Original through save/reload pipeline | Matched | Detect serialization or conversion changes |
| Edited candidate | Selected edit | Matched to its baseline | Estimate the effect of the weights |

Optional negative controls include a random direction edit matched for layer, changed-coordinate count, and norm. Treat these as additional experimental arms, with their own seeds and budget; do not select the most favorable random control after looking at the test.

Changing model mode, precision, tokenizer, prompt wording, generation budget, or output processing only for the edited arm invalidates the intended paired comparison. A secondary experiment may deliberately change one of these, but must identify the changed factor.

### Editor intervention

Allow inspection of approved weights and approved development forward passes. Allow deterministic edits to existing tensors. Keep the architecture, tokenizer, and evaluation harness fixed. Prohibit gradient computation, optimizer updates, adapters trained on examples, distillation, and hidden training loops for the direct-edit arm.

Record every candidate, including rejected ones. Start each candidate from the immutable original checkpoint unless a composition is explicitly specified. Record the proposal's input hashes, rationale, tensor names, coordinates or operators, before/after hashes, and validation result. Preserve the first unsuccessful attempt as faithfully as a later successful one.

A patch rationale is scientific provenance. Development conversations and code-generation instructions are separate working material and do not belong in the public source tree.

### Budget

Count calls, requested sequences, completed sequences, generated tokens, wall time, and any billed usage separately. One batched evaluator call can contain many generations; calling it one example would materially understate the search budget.

The template's generated-token cap and maximum edited fraction are proposed pilot limits. They are not limits taken from the motivating study. Its generic patch operations also need an execution policy matching the original intervention restrictions before a closer reproduction is claimed. Revise these before freezing the protocol if the aim is closer methodological comparability. Stop when a budget is exhausted; never silently expand it after viewing results.

### Scoring

Use a pinned mathematical answer parser and verifier for MATH-500. Verify its handling of equivalent forms, missing answers, multiple boxed answers, truncated outputs, and malformed output using independently prepared examples. Preserve raw generations and the extracted answer for audit. Do not substitute an unvalidated string matcher or an unpinned model judge.

Use the published IFEval evaluator and clearly label strict versus loose, and instruction versus prompt-level scores. Score GSM8K with a documented final-answer extraction policy. Report the average per-generation correctness; do not relabel a best-of-four success rate as pass@1.

The primary metric gives each problem equal weight and averages its scheduled samples; schedule the same number of samples for every problem in a benchmark. This then equals average correctness across all scheduled item/seed pairs. The analysis utility also reports the case-weighted mean so uneven schedules remain visible. Record runtime failures and missing outputs; do not remove them from the denominator. Also report benchmark correctness under its declared truncation policy, token-limit rate, repetitive-output diagnostics, generated-token count, response duration, and completion within a predeclared practical budget. A mathematically correct truncated answer and a completed answer within a practical budget can be different metrics; specify each policy before running.

Pair conditions on the same item/seed schedule. Repeated samples of one problem are correlated. Bootstrap paired differences by problem, retaining all of its samples together. Show the full per-item record, effect size, uncertainty interval, and regressions as well as an aggregate score. A seed improves traceability without guaranteeing identical behavior across engines or hardware.

Native duration is not browser time-to-first-token, and tokens/second cannot stand in for correctness. Do not combine unlike runtime measurements into one unlabeled leaderboard.

## Track B: browser deployment

Use BrowserLLM as a related project at source revision `b39187b1aff1f218493622d7fcf3cd5d24773b41`. The current audit confirms Qwen3.5-0.8B artifacts and WebLLM 0.2.86 support, but no task-correctness grading in its existing workloads. See [browser-audit.md](browser-audit.md). Do not modify its shared repository as part of this project's initial setup.

First test unchanged versus edited checkpoints under the same native precision. Then convert both with the same converter and quantization settings, record artifact hashes, and repeat the quality comparison. Small edits can disappear or change under quantization; the preconversion result is insufficient evidence.

Before a full benchmark, require an identity conversion to load, tokenize, generate, and export a result through the selected browser runtime. Record its actual output behavior and conversion error. Existing Qwen3.5 support is encouraging, but does not prove that a newly converted edited checkpoint matches the existing compiled configuration. Resolve that compatibility on the same architecture before proceeding; do not silently replace the model and retain a direct-replication label.

For a first application benchmark, structured extraction is a practical candidate because required fields can be graded against explicit ground truth. Define schema validity and semantic field accuracy separately. Freeze development, validation, and withheld task templates; generated examples from a shared template do not create independent generalization evidence.

A proposed practical metric is:

`100 × tasks correct within every required budget / all scheduled tasks`.

Publish accuracy and each budget measure alongside that composite. Record the browser version, operating system, hardware, runtime, precision, foreground state, warm/cold state, and independent session. Alternate base/edited execution order and run independent sessions to reduce order, caching, and thermal effects.

## Decision rules

- If decoding alone closes the gap, report it. A useful result need not favor weight editing.
- If improvements disappear with quantization, report a native-only effect.
- If task quality improves but a prespecified resource budget is exceeded, do not call the browser deployment suitable under that budget.
- If the model no longer loops but held-out accuracy does not improve, report the narrower behavioral change.
- If the edited model regresses on other tasks, show the tradeoff.
- If no edit improves the final metric, release the negative result and full search budget.

The project will support a claim only after its real checkpoint and independently evaluated outputs exist. The mechanics example is never evidence of language-model improvement.

## Execution contracts

Use the [version 2 profiles](protocol-profiles.md) to distinguish artifact preparation, development smoke runs, fixed replay evaluation, fresh-editor search, frozen final evaluation, and browser evaluation. Preserve the original proposed settings until an explicit protocol revision is approved and recorded; null execution fields do not supply defaults. Each benchmark declares its own item and seed schedule.

Retain [raw run records](evidence-records.md) separately from compact correctness records. Build and fixture-validation success do not establish real-model validation or an executed experiment. Diagnostic backend availability also establishes neither. Independent grades must identify their evaluator and the exact raw record they grade, and failed attempts remain part of the budget and evidence.
