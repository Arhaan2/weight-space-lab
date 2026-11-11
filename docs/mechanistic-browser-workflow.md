# Mechanistic editing with one joint browser comparison

The objective is to let Astra inspect model weights and behavior, make a targeted
weight edit, and test whether that edit improves **IFEval instruction compliance
and browser resource usage on the same requests**. IFEval is the only accuracy
benchmark in this version. Browser LLM Bench supplies the runtime measurement;
there is no separate autumn-leaves workload and no required four- or five-edit
sweep. More accuracy benchmarks can be added after this workflow is evaluated.

The first actual paired study completed on November 11, 2026: strict compliance
fell from 16/32 to 15/32 and output tokens rose 1.25%. It did not demonstrate a
joint improvement. See the [result, exact edit and limitations](../reports/mechanistic-browser-v1/README.md).
Historical V1, V2, R4 and resource results remain unchanged.

The default schedule is one baseline browser session, adaptive diagnostic tools,
one submitted candidate, one audited conversion, and one candidate browser
session. Each browser session has one excluded warm-up and one generation per
scheduled IFEval prompt. A 32-prompt workload therefore uses 64 scored requests
and two warm-ups. The existing official grader scores saved answers without
generating again. Activations use separately accounted CPU diagnostic forwards.

## What changed

The new [controller](../weight_space_lab/mechanistic_study.py) uses the
[single-pair configuration](../configs/ifeval_mechanistic_browser.json) and a
separate [mechanistic policy](../configs/mechanistic_candidate_policy.json).
The policy is resolved against the actual inventory before freezing: Astra can
inspect any inventoried tensor through bounded metadata, raw-slice and statistics
tools. Editing is allowed in all 24 MLP down-projection matrices, with arbitrary
column IDs, rather than just two preselected contiguous blocks. The default
aggregate intervention budget is 131,072 source coordinates. Embeddings, output
head, tokenizer, architecture, quantization and decoding remain fixed.

Directional operations support strengths from 0 through 6:

```text
W'[:, C] = W[:, C] - strength * u * (u.T @ W[:, C])
```

The implementation normalizes `u`, applies the operation in binary64, rounds
directly to BF16, and independently checks the saved source checkpoint. Strength
6 reproduces the *form* of the published p50c counteraction; it is not a claim
that a newly derived direction or candidate reproduces the published mechanism.
Column scaling remains available. The original policies have not been relaxed.

The diagnostic worker captures selected residual outputs and the inputs to MLP
down-projections on matched reference/failure continuations. A contrast tool
derives a unit residual direction and ranks columns by their activation contrast
times their projection onto that direction. Astra can inspect results and request
another bounded diagnostic before making its final submission. These tools do
not run gradients, training, or arbitrary editor-supplied Python.

**Diagnostic provenance:** captures come from the exact source checkpoint using
the pinned CPU Transformers runtime. They are not activations from the quantized
WebGPU executable. Browser evaluation is the authority for deployed performance;
the conversion audit separately records quantization changes. The activation
worker is limited to 180 seconds and 8 GiB sampled RSS per call by default;
sampling is not a hard allocator ceiling. At most 16 activation requests are
allowed in the default study. See [inspection details](mechanistic-inspection.md).

## Prepare without inference

Use a new private directory under ignored `artifacts/`. Supply the verified
pristine source, its paired browser-conversion manifest, the existing task-mode
browser checkout with pinned dependencies, the exact IFEval item file, and the
official evaluator installation. These are local inputs; preparation downloads
nothing and does not load an inference engine.

```sh
python -m weight_space_lab.mechanistic_study prepare \
  --root artifacts/ifeval-mechanistic-browser-v1 \
  --source-model /absolute/path/to/pristine-model \
  --baseline-manifest /absolute/path/to/conversion/artifacts.json \
  --baseline-config /absolute/path/to/baseline-config.json \
  --benchmark /absolute/path/to/task-mode-browser-bench \
  --items /absolute/path/to/development-items.json \
  --official-ifeval-root /absolute/path/to/official-ifeval
```

The paths above are placeholders. The baseline config must bind the same source,
deployed artifact, prompt text, item IDs and exact rendered input-token counts.
For the existing workload, reuse its saved verified baseline template. For a new
IFEval development set, first run the existing browser checkout's
`scripts/count-task-tokens.mjs TOKENIZER CHAT_CONFIG ITEMS OUTPUT` and construct
the matching template; this performs tokenization only. The runtime also checks
observed rendered input-token counts before dispatching a scored answer.

Preparation copies and instruments the browser harness, builds the task client,
freezes workload/policy/runtime identities, and initializes a new ledger. It
never resets a historical budget. All prompts in this workflow are explicitly
development prompts because the editor can see their baseline responses. A
future untouched confirmation set needs a separately designed evaluation.

## One adaptive Astra session

Run the baseline once and export its packet:

```sh
python -m weight_space_lab.mechanistic_study baseline --root STUDY
python -m weight_space_lab.mechanistic_study packet --root STUDY --output STUDY/editor-packet.json
```

The packet contains the objective, actual baseline responses and grades, joint
resource results, resolved policy, source identity, tool schema and remaining
accounting. Give it to the requested `gpt-6-astra` session with access to the
controller's `tool` command. This is a continuing diagnostic session, not a single
guess from static weight summaries. The local command boundary does not establish
OS isolation or immutable provider version; those remain explicitly unverified.

Each tool request binds the study SHA:

```json
{
  "study_sha256": "COPY_FROM_EDITOR_PACKET",
  "tool": "inspect",
  "arguments": {
    "op": "metadata",
    "tensors": ["model.language_model.layers.15.mlp.down_proj.weight"]
  }
}
```

```sh
python -m weight_space_lab.mechanistic_study tool --root STUDY --request REQUEST.json
```

For `capture`, use the exact request schema in
[mechanistic-inspection.md](mechanistic-inspection.md), including the frozen
source SHA, selected layers and matched reference/failure texts. For `derive`,
the arguments are `{"capture_id":"RETURNED_ACTION_ID","top_k":128}`. Only a
completed capture from the same study is accepted. The result returns an action
ID and its hash-bound direction/ranking data.

To turn selected diagnostic directions into a proposal without manually copying
1,024 direction values, use the pure algebraic helper:

```sh
python -m weight_space_lab.mechanistic_tools propose \
  --model ORIGINAL_MODEL --policy STUDY/policy.json \
  --directions STUDY/diagnostics/DERIVATION_ACTION_ID/directions.json \
  --selections selections.json --output STUDY/directional-proposal.json
```

`selections.json` is an array such as
`[{"layer":15,"columns":[0,17,294],"strength":6}]`; these are illustrative
indices, not a recommended intervention. Astra should choose columns and doses
using the actual evidence. The helper validates provenance and records a sidecar;
it does not run the target model or create another candidate slot.

Submit one final proposal and a pre-evaluation rationale:

```sh
python -m weight_space_lab.mechanistic_study submit --root STUDY \
  --proposal STUDY/directional-proposal.json --rationale rationale.txt \
  --editor-provenance editor-provenance.json
python -m weight_space_lab.mechanistic_study convert --root STUDY \
  --converter-python /absolute/path/to/pinned-converter/python
python -m weight_space_lab.mechanistic_study candidate --root STUDY
python -m weight_space_lab.mechanistic_study report --root STUDY --output joint-comparison.json
```

The candidate always starts from pristine. Submission, conversion and browser
attempts are reserved before work and cannot be silently replayed. Invalid
submissions and failed attempts retain their spent capacity. An interrupted
active action blocks automatic continuation. No extra candidate is substituted
because its measured result is worse. If quantization erases the edit, the
conversion outcome is recorded and candidate evaluation stops before consuming a
browser run. Diagnostic tools close after submission.

## Accuracy and usage on the same workload

The [joint adapter](../weight_space_lab/joint_browser.py) enables Browser LLM
Bench's native collector while the exact IFEval prompts run in WebLLM. It joins
item IDs, output hashes, token counts, response boundaries and official grades.
Warm-ups are excluded from scored totals. Missing and failed scheduled prompts
remain in quality denominators. Failed-response cost is retained; missing cost
makes complete-workload cost ratios unavailable rather than artificially low.

Per-request resource summaries use only valid measured samples/intervals inside
that request. They never copy a session average into every row. Short requests
can have insufficient telemetry coverage, which is reported as unavailable.
CPU/GPU activity is reported with coverage and duration; it is not FLOPs. Native
footprint is sampled process accounting, not globally deduplicated RAM. Buffer
descriptors and WASM address space are not physical VRAM. Host energy, when
available with complete interval coverage, remains host-level and is never
presented as model-only energy. Missing power remains null.

The report includes strict/loose compliance, tokens, latency, memory and measured
activity on the same requests; paired changed outcomes; and total workload
tokens/response time per strictly compliant answer. All scheduled work belongs
in the numerator, including incorrect answers. A zero compliant count or missing
cost produces null ratios. Joint improvement requires evidence across quality
and the chosen resource metric; no scalar utility weights are invented.

One session per arm is a descriptive paired pilot. It avoids unnecessary repeated
full runs, but cannot establish resource equivalence, causal attribution across
changing host conditions, or robustness across devices. Preserve that limitation
when interpreting a positive or negative result.

## Website and release

The website's Harness view explains the new workflow and keeps completed
historical comparisons separate. It starts with an explicit not-run state.
The updater can import the canonical joint report when an experiment finishes;
only public numeric and identity fields belong in the website. Raw prompts,
responses, activation arrays and editor packets remain in the private study.
See [site publication and verification](results-site.md).

The implementation checks exercise actual synthetic directional checkpoint edits,
activation hooks and capture limits, telemetry/grade alignment, failure costs,
single-use state, and the compiled browser integration. They are engineering
validation, not new model-performance measurements.

## First-run startup amendment

The first candidate browser process closed before any model load or generation.
The original failure, zero-call ledger and study state were preserved. A separate
[explicit pre-load continuation](../scripts/run_mechanistic_preload_continuation.py)
verified zero prior target calls, froze the same candidate/workload/runtime, and
completed its first model evaluation. This amends the original no-retry launch
protocol without repeating target inference. Its gate refuses existing load
grants, target events, nonzero/unknown counts, ledger attempts, incomplete cleanup,
changed evidence or a consumed continuation. The public report retains the failed
launch's 6.120-second supervisor cost and separate browser/model counts.
