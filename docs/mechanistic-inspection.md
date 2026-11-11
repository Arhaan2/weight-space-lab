# Weight and activation inspection

This describes the historical bounded v1 tools. The recommended
[research workflow](research-browser-workflow.md) adds persistent Python,
arbitrary internal edits and general original/edited forward probes.

The new mechanistic tool path gives Astra actual bounded weight slices, selected
teacher-forced activations, and directional edits. The original V1/V2/R4 policies
and results remain historical. This path does not silently extend their budgets
or turn their scaling edits into directional experiments.

The objective remains **one IFEval workload evaluated for both instruction
compliance and WebLLM/browser resource usage**. CPU diagnostic forwards help
explain a failure and choose a candidate. They are recorded separately and never
substitute for the browser measurement. The main comparison is one original
checkpoint and one selected candidate, rather than a required four- or five-edit
sweep. Additional accuracy benchmarks are outside the current workflow.

## Available tools

`weight_space_lab.mechanistic_tools` supplies a Python interface and a module
CLI. All JSON proposals are declarative; no editor-provided Python is evaluated.

| Tool | Function | Evidence |
| --- | --- | --- |
| Inspect | `inspect_weights(model, policy, request, exposure_log)` | Actual metadata, bounded raw slices or statistics, source identity and durable exposure journal |
| Capture | `capture_activations(model_path, request, output_dir, limits=None)` | Source-model residual and MLP activations for explicitly supplied reference/failure text pairs |
| Derive | `derive_directions(model_path, captures_dir, output_path, policy, top_k=128)` | Unit residual contrast and ranked neuron contributions for each captured layer |
| Propose | `build_direction_proposal(model_path, policy, diagnostics_path, selections, output_path)` | Existing checkpoint proposal schema plus a hashed evidence sidecar |
| Apply | `apply_proposal(model, policy, proposal, output, exposure_log)` | Independently verified saved candidate, without target inference |

Before opening the editor, `policy_for_inventory(template, inventory)` expands
read access to every tensor in the exact checkpoint inventory, including
attention matrices, MLP gate/up projections, norms, embeddings and buffers. The
controller freezes that resolved policy. This broadens inspection only: the
mutation allowlist remains the MLP down-projections. There is no wildcard that
admits a later unrecorded tensor, and all reads retain the value/scan budgets.

The run controller owns cumulative reservations for diagnostic calls, forwards,
proposals and browser evaluation. These functions enforce per-operation bounds;
calling the standalone CLI is not a substitute for the run ledger. The content
hashes bind saved artifacts but are not cryptographic signatures or proof that
an editor could not access other files.

## What the activations mean

The implemented capture is a **CPU Transformers source-model diagnostic**, not
an instrumentation claim about WebGPU's quantized internal activations. The
optional runtime uses the same pinned `torch==2.10.0`, `transformers==5.3.0` and
supporting packages as the native backend. The loader preserves the checkpoint's
stored BF16/F32 values and verifies loaded tensors before forwarding.

For Qwen3.5, a hook on the decoder block captures the residual stream **after**
that layer's MLP addition. A pre-hook on `mlp.down_proj` captures its input: the
post-gating MLP neuron activations, before multiplication by the output weights.
The text backbone is called directly, with `use_cache=False`, under
`torch.inference_mode()`. No vocabulary-sized logits, generation, backward pass,
optimizer, training loop, network download or remote code is requested.

The hook locations were checked against the pinned
[Transformers v5.3.0 Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
The actual inference runtime is optional and is imported only by an explicit
capture call. Fixture tests use mocked hooks and synthetic arrays; they are not
evidence of a real Qwen capture or accuracy gain.

Source diagnostic evidence can suggest an intervention, but quantization can
change its effects. The converted candidate still needs effective quantization
audits and the same browser quality/resource workload as the baseline.

## Capture request

Provide exact rendered text, including any desired chat markers and
continuations. Tokenization uses `add_special_tokens=False`; the tool does not
invent or change a chat template. Each pair has exactly one `reference` and one
`failure`. Those labels are hypotheses, not evaluator outcomes.

```json
{
  "schema_version": 1,
  "source_sha256": "REPLACE_WITH_THE_FROZEN_SOURCE_MANIFEST_IDENTITY",
  "layers": [15, 19],
  "tail_tokens": 24,
  "max_tokens": 512,
  "cpu_threads": 4,
  "texts": [
    {
      "id": "case-1-reference",
      "pair_id": "case-1",
      "condition": "reference",
      "text": "Exact rendered prompt and productive continuation",
      "token_span": null
    },
    {
      "id": "case-1-failure",
      "pair_id": "case-1",
      "condition": "failure",
      "text": "Exact matched prompt and failing continuation",
      "token_span": null
    }
  ]
}
```

The placeholder hash intentionally fails validation. Use the source identity
from the frozen experiment. `token_span` may be an explicit zero-based
half-open `[start,end]` token interval. If null, the last `tail_tokens` positions
are captured. Requests exceeding the token ceiling are rejected, never silently
truncated. Saved token IDs and spans make each selection inspectable.

Default maximums are 16 texts, 24 selected layers, 512 tokens per text, 32 captured
positions per text and 16,777,216 saved activation elements in total. The request
contains at most 65,536 text bytes. The checkpoint is capped at 4 GiB. A separate
worker has a 180-second wall allowance and an 8 GiB sampled process-RSS ceiling;
the controller kills its process group on timeout, memory overrun, unavailable
memory telemetry or excessive worker log output. These limits may be tightened
per request, not raised through the request.

RSS sampling is a stop threshold, not an instantaneous allocator guarantee. A
short overshoot between samples remains possible. Its measured memory is not
browser RAM and must not enter the browser resource comparison. Capture success
requires process exit, original-source identity, and verified output hashes.

The output directory is new and outside the original checkpoint. It contains:

- `request.json` and `limits.json`: exact private inputs and bounds.
- `activations.npz`: selected per-token residual/MLP arrays, with no pickle data.
- `manifest.json`: `{payload, sha256}`, binding source, request, array bytes,
  actual token selections, backend, and loaded-model verification.
- `supervision.json`: hash-bound wall time, sampled process peak and exit status.

Prompt/activation artifacts are research inputs, not public website downloads.
The run controller should expose only the intended development observations to
the editor and publish sanitized aggregate evidence.

## Directions and edits

For each layer, token means are computed inside the explicit captured spans.
Matched pairs receive equal weight. The direction is the normalized average
failure-minus-reference residual contrast. For neuron `j`, its writer score is:

```text
score[j] = mean_pairs(MLP_failure[j] - MLP_reference[j]) * dot(u, W[:, j])
```

The saved result gives the unit direction, ranked column IDs, scores and the
number of pairs with positive attribution. Ranking is by contribution to the
observed contrast, not weight magnitude or neighboring column index. Zero
contrasts, nonfinite arrays, source/hash mismatches and incomplete pairs are
rejected. These scores are associations; causal checks and browser evaluation
are still needed before claiming a mechanism or benefit.

The new `configs/mechanistic_candidate_policy.json` allows editing the 24 text-layer MLP
down-projections. The resolved policy permits reading all inventoried tensors.
Unlike the previous family, selected columns need not be
contiguous and the sites need not be layers 15/19. A direction can have strength
from 0 through 6. The total support remains at most 131,072 coordinates, with
at most 24 operations, a stored changed-coordinate fraction at most 0.001, and
a maximum absolute stored change of 4. Original tensor shapes and other assets
must remain unchanged. Original policies are not modified.

```json
[
  {"layer": 15, "columns": [0, 17, 294], "strength": 2.0},
  {"layer": 19, "columns": [1, 11, 343], "strength": 2.0}
]
```

These are illustrative indices, not a proposed or tested intervention. Each
selected column undergoes `W'[:,C] = W[:,C] - strength*u*(uᵀW[:,C])`. At strength
one this removes the component; above one it counteracts it. Strength six can
express the published p50c dose, but matching its effect also requires the
correct direction, support, source values and deployed runtime.

The proposal builder writes the existing validated proposal schema and an
adjacent `.evidence.json` sidecar linking the direction and capture hashes. Raw
editor-authored proposals can also use the existing directional operation. The
host builds an independent copy and runs the saved-byte verifier; source model
bytes are never overwritten. Numeric arithmetic and BF16 rounding are the same
audited checkpoint implementation used elsewhere in this project.

## Standalone commands

Use the experiment controller for accounted runs. The underlying tools can be
checked independently with explicit prepared paths:

```sh
python3 -m weight_space_lab.mechanistic_tools resolve-policy --model /absolute/model --policy configs/mechanistic_candidate_policy.json --output /absolute/resolved-policy.json
python3 -m weight_space_lab.mechanistic_tools inspect --model /absolute/model --policy /absolute/resolved-policy.json --request /absolute/inspection.json --exposure-log /absolute/exposure.jsonl
python3 -m weight_space_lab.mechanistic_tools capture --model /absolute/model --request /absolute/capture-request.json --output /absolute/new-capture
python3 -m weight_space_lab.mechanistic_tools derive --model /absolute/model --policy /absolute/resolved-policy.json --capture /absolute/new-capture --output /absolute/directions.json
python3 -m weight_space_lab.mechanistic_tools propose --model /absolute/model --policy /absolute/resolved-policy.json --directions /absolute/directions.json --selections /absolute/selections.json --output /absolute/proposal.json
python3 -m weight_space_lab.mechanistic_tools apply --model /absolute/model --policy /absolute/resolved-policy.json --proposal /absolute/proposal.json --output /absolute/new-candidate --exposure-log /absolute/exposure.jsonl
```

An inspection JSON uses the existing `metadata`, `slice` or `statistics` schema.
For example, `{"op":"slice","tensor":"model.language_model.layers.15.mlp.down_proj.weight","slices":[[0,16],[0,16]]}` returns the actual 16×16 values and journals the disclosure.

Verification for this implementation runs without a checkpoint or optional
inference packages:

```sh
python3 -m unittest discover -s tests -p 'test_mechanistic_tools.py' -v
```
