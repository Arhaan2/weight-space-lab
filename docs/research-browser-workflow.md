# Open weight research, one final browser comparison

This is the recommended harness for the next Astra experiment. It follows the
[original Experiment 8 tool contract](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/turns/t001/prompt.txt):
persistent Python over weights and metered diagnostic forwards. The final
objective remains **IFEval accuracy and WebLLM resource usage on the same
responses**. IFEval is the only accuracy benchmark.

The [controller](../weight_space_lab/research_study.py) and
[configuration](../configs/ifeval_research_browser.json) replace the restricted
contrast-and-ranking workflow for new studies. The previous controller and its
[completed negative result](../reports/mechanistic-browser-v1/README.md) remain
historical records. This implementation update has no model-performance result.

## Research before one frozen comparison

1. Prepare a new study, freezing source/deployment identities, practice/final
   items, runtime, settings and budgets. Preparation runs no target inference.
2. Astra writes analyses, edits internal weights, compares original and edited
   behavior, and keeps/restores named candidate snapshots on practice data.
3. Freeze one snapshot and its rationale. Stop the kernel and close all Python,
   forward, snapshot and restore tools before either final browser arm runs.
4. Materialize and audit that checkpoint, then convert once with the pinned MLC
   toolchain. Audit actual deployed changes, including fusion and quantization.
5. Run one pristine browser session and one candidate session. Each has one
   excluded warm-up and one generation per final IFEval item. Grade and measure
   these same saved responses.

For 32 final prompts, this is 64 scored generations and two warm-ups. Diagnostic
generations are additional **development work**, reported separately, never
browser resource measurements. Final results cannot select another candidate.
Failed attempts retain their evidence and spend capacity; no automatic retries.

Practice uses `development`/`validation` items; final evaluation uses `test` items.
Preparation rejects overlap in IDs, source rows, normalized prompts and IFEval
instruction-combination families. Final texts/targets are excluded from the
editor packet and kernel read permissions. Probes containing known final prompts
or IDs are rejected. Withholding applies to this session; it does not establish
that public data were never seen in earlier studies or training.

## Weight-editing API

| Object or function | Behavior |
| --- | --- |
| `NAMES` | All inventoried tensor names, including read-only tensors |
| `load_original(name)` | Independent float32 CPU Torch copy of pristine values |
| `load(name)` | Current float32 edit, otherwise pristine values |
| `save_edit(name, tensor)` | Direct replacement preserving the original shape |
| `drop_edit(name)` / `edited()` | Remove an edit / list active edited tensors |
| `tok` | `tokenizers.Tokenizer`; `tok.encode(text).ids` yields token IDs |
| `np`, `torch`, `WORK`, `READ_PATHS` | Scientific tools and explicit evidence paths |

Python variables persist between calls. SciPy, scikit-learn, pandas, SymPy and
NetworkX can be imported when installed. The harness imposes no direction
estimator, neuron score, repetition hypothesis, chosen layer, contiguous-column
restriction, rank constraint, strength-six ceiling or 131,072-coordinate cap.
Astra can write head/neuron ablations, directional updates, decompositions,
norm-matched random controls and other justified multi-tensor edits itself.

All floating tensors inside the 24 text transformer blocks are editable,
including attention, linear attention, MLPs and internal norms. As in the
original, embeddings, output head, final norm and whole-tensor scalar-only
changes are excluded. Architecture, shapes, stored dtypes, tokenizer and final
decoding controls stay fixed. Training, gradients, optimizers, and implementing
target forwards inside the notebook are outside the scientific contract.

Submitted float32 values persist in notebooks/snapshots. Materialization casts
to the original stored dtype, including BF16, and independently audits saved
bytes. Source files stay unchanged. Restore updates active edits without
restarting Python; previously assigned notebook variables retain their values.

## Diagnostic forward tools

Both request types accept `model: "original"`, `"edited"`, or `"both"`. Edited
probes bind an immutable snapshot before running. The tools permit dose-response,
layer-isolation, random-control and transfer experiments without rerunning the
final browser benchmark.

Teacher forcing accepts arbitrary raw `texts` or `chats`, with explicit thinking
template behavior. It can return token log probabilities, top-k alternatives,
hidden states, gated MLP activations and full-attention probabilities. It does
not require matched pairs, fixed tail windows or a supplied contrast procedure.

```json
{
  "type": "teacher_forced", "model": "both",
  "texts": ["An explicitly rendered practice context and continuation"],
  "return": {
    "token_logprobs": true, "topk": 10,
    "hidden_states": [0, 15, 16, 19, 20, 24],
    "mlp_act": [15, 19], "attn_layers": [15, 19]
  }
}
```

Hidden state 0 is embeddings; 1–23 are block outputs; 24 is the final normalized
state. MLP `i` is the input to block `i`'s down projection. Ordinary full attention
is available at blocks 3, 7, 11, 15, 19 and 23; linear-attention states are not
substituted. Arrays identify arm/text. Log-probability position `i` predicts input
token `i+1`; the first input token is unscored. No silent input truncation.

Generation accepts practice IDs and/or custom prompts. Known IFEval items receive
official strict/loose grades; custom prompts have no invented accuracy grade.
Paired arms use the same per-item/sample seeds and report paired accuracy change,
standard error, regressions, rescues, lengths and truncations.

```json
{
  "type": "generate", "model": "both",
  "problems": ["REPLACE_WITH_AN_ID_FROM_THE_PRACTICE_PACKET"],
  "n": 1, "max_tokens": 1024,
  "settings": {"temperature": 0, "enable_thinking": false},
  "activations": {"layers": [16, 20], "mlp": true, "max_answer_tokens": 512}
}
```

Optional generated-answer activations use a teacher-forced replay of the prompt
and a bounded answer prefix. Replays, tokens and actual model calls are counted
separately; they are not free or additional generations. Outputs persist before
grading. Failures retain partial progress, errors and logs. Temporary model
copies are removed after probes; snapshots and audits permit reconstruction.

## Budgets, runtime and isolation

Defaults allow **up to 300 Python calls and 50 forward attempts**. These are
ceilings, not targets to exhaust. Unlike original published accounting, failed
attempts spend capacity. A paired request uses one tool attempt while both
arms' actual sequences and model calls are recorded.

Python calls have a 900-second wall limit, 8 GiB sampled RSS ceiling and 10,000
output-character limit. The session also has idle/lifetime bounds. Restarts are
explicit, use new control directories and retain disk edits; lost variables are
never silently replaced. At most eight starts are allowed per study.

Forward tools support up to 1,024 teacher-forced texts, 32,768 input tokens per
text, 524,288 total teacher-forced tokens, 512 generated sequences across arms,
32,768 new tokens per generation, and 6 GiB saved arrays. Attention capture allows
two texts of at most 2,048 tokens. Default generation reservations allow
1,048,576 tokens including prompts, answers, both arms and activation replays.
The worker has a 3,600-second wall limit and 8 GiB sampled RSS ceiling. Limits are
configurable within tool maxima before preparation; original hardware capacity
is not reproduced.

Diagnostics use pinned **CPU BF16 source weights**, preserving stored F32 tensors
where applicable. Final evaluation uses **q4f16_1 WebLLM/WebGPU**. Deployment
audits cover fused Q/K/V, gate/up projections, convolution and norm offsets, and
reject changes outside source-supported coordinates/quantization groups.

The notebook uses macOS Seatbelt: weights, packages and permitted probe data are
read-only; only its workspace is writable. Network, child processes and GPU
access are not granted. Unsupported hosts have no unrestricted fallback. Known
training/target-execution APIs are guarded, but general Python cannot prove that
arbitrary algebra is not a hand-written optimizer or simulated forward pass.
Code logs still require scientific review. Kernel confinement does not prove
outer-editor isolation or authenticate the Astra backend version.

## Commands

Use a dedicated environment with existing local model, browser and official
grader assets. The analysis extra adds scientific libraries; source inference
remains pinned:

```sh
python -m pip install -e '.[native-cpu,grading,research]'
python -m weight_space_lab.research_study prepare \
  --root artifacts/ifeval-research-browser-v2 \
  --source-model /absolute/pristine-model \
  --baseline-manifest /absolute/conversion/artifacts.json \
  --baseline-config /absolute/final-baseline-template.json \
  --benchmark /absolute/task-mode-browser-bench \
  --practice-items /absolute/ifeval-practice.json \
  --final-items /absolute/ifeval-final.json \
  --official-ifeval-root /absolute/official-ifeval \
  --probe-python /absolute/pinned-native-cpu/bin/python
```

Paths are placeholders. Final items need matching rendered token counts in the
baseline template; the browser's existing token-count script prepares these
without inference. Final settings remain thinking off, temperature 0, seed 42,
context 4,096 and maximum 1,024 output tokens. Diagnostic settings may vary for
controlled probes and are recorded per request.

```sh
python -m weight_space_lab.research_study packet --root STUDY --output STUDY/editor-packet.json
python -m weight_space_lab.research_study kernel-start --root STUDY \
  --python-executable /absolute/scientific-environment/bin/python
python -m weight_space_lab.research_study python --root STUDY --code-file analysis.py
python -m weight_space_lab.research_study forward --root STUDY --request probe.json
python -m weight_space_lab.research_study snapshot --root STUDY --name promising --evidence 'Observed evidence'
python -m weight_space_lab.research_study restore --root STUDY --name promising
```

Give Astra the practice packet, API and request schemas. Ask it to form hypotheses
and obtain causal evidence, including relevant strengths, isolated components,
controls and unused practice examples. These are available research choices,
not prescribed edits or mandatory repeated benchmark runs.

```sh
python -m weight_space_lab.research_study freeze --root STUDY --name promising \
  --rationale-file rationale.txt --editor-provenance editor-provenance.json
python -m weight_space_lab.research_study convert --root STUDY \
  --converter-python /absolute/pinned-converter/bin/python
python -m weight_space_lab.research_study compare --root STUDY
python -m weight_space_lab.research_study report --root STUDY --output joint-comparison.json
```

`compare` runs the final baseline and candidate once, in order. `baseline` and
`candidate` expose those same single-use slots for staged execution. Existing
arms are never replayed. If conversion erases the edit, the outcome is recorded
and neither browser arm runs. `kernel-stop` is always available for cleanup and
cannot reopen development. A failed freeze also closes development.

The [joint measurement adapter](joint-browser-measurement.md) reports accuracy,
tokens, response time, available CPU/GPU/memory telemetry and cost per compliant
answer from the same requests. Missing energy stays null. One final pair is a
descriptive comparison, not replicated evidence of a causal speedup. Development
cost is reported separately from final inference cost.
