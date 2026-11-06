# Real candidate-checkpoint editing

Phase 6 adds a bounded editor interface for complete local Safetensors model
directories. It resolves the stored tensor names and shard index, changes
declared coordinates in their original storage dtype, and reopens the saved
candidate before accepting it. Architecture configuration, tokenizer assets,
chat templates, shard headers, unrelated files, and nonfloating buffers remain
byte-identical. This is candidate-editing infrastructure, not an Astra search
or evidence of improved model capability.

## Original identity and candidate composition

`CheckpointEditor(model, policy, exposure_log=...)` binds the original model's
complete file inventory. A proposal's `base_sha256` is the canonical SHA256 of
that inventory's `files` object, not the old synthetic-array state hash. Every
proposal begins from that same immutable original. Calling `propose` again
never uses the previous candidate as its starting point.

An `independent` proposal contains zero or one operation. A `sequential`
proposal explicitly composes its operations in array order, starting from the
original and rounding after each operation. Parent-candidate chaining is not
supported. To compose edits, declare them together in one sequential proposal.
Budgets apply to the complete composition relative to the original, including
its union of targeted coordinates.

```json
{
  "schema_version": 1,
  "base_sha256": "<canonical SHA256 of the original inventory files>",
  "provenance": "engineering_edit",
  "composition": "independent",
  "operations": [{
    "op": "scale_columns",
    "tensor": "model.language_model.layers.15.mlp.down_proj.weight",
    "columns": [0],
    "factor": 0.99
  }]
}
```

`provenance` is `engineering_edit` or `editor_proposal`. This declaration alone
does not authenticate a model identity, prove that an editor made the proposal,
or establish that the editor had no access to held-out information. Phase 7
must enforce those access boundaries.

## Supported operations and arithmetic

The operation names and algebra match the original bounded array mechanics:

| Operation | Required arguments | Effect |
| --- | --- | --- |
| `scale_columns` | `tensor`, unique `columns`, finite `factor` | Multiply every row of the declared columns by the factor. |
| `remove_direction_from_columns` | `tensor`, unique `columns`, finite nonzero `direction`, finite `strength` | Normalize the row-length direction and subtract `strength * u * (uᵀ W)` from those columns. |

Only two-dimensional F16, BF16, F32, and F64 tensors accept arithmetic.
Direction vectors must have exactly the matrix's row count. There are no
implicit transposes, tensor aliases, broadcasts to additional columns,
trainable parameters, or architecture transformations. Nonfloating buffers
are preserved and cannot be arithmetic targets. Signed and unsigned integer
and Boolean slices retain their exact values. F8_E4M3 and F8_E5M2 tensors support read-only
inspection and finite checks; float8 arithmetic is unsupported.

The working precision is binary64. Every operation decodes only its bounded
selection, performs the algebra in binary64, and rounds back to the original
stored dtype before the next operation. Rounding is nearest with ties to even;
BF16 rounds directly from binary64 rather than through an intermediate F32
cast. This avoids double rounding. Underflow may produce subnormals or signed
zero; nonfinite results and storage overflow are rejected. The audit records
the precision contract. Floating-point reductions follow the recorded NumPy
implementation; this is not a promise of bitwise-identical arithmetic across
all libraries and hardware. The audit records the NumPy runtime version.

The adapter never converts the complete checkpoint into a generic NumPy state.
It reads typed tensor spans and bounded selections, copies original files as
bytes, and writes only selected tensor coordinates. Full-file hashes and
bounded verification scans still read the checkpoint for integrity checking.

## Policy enforcement and saved verification

[candidate_policy.json](../configs/candidate_policy.json) is a concrete example
for the two declared Qwen MLP tensors. It is separate from a proposal so the
proposal cannot grant itself access. The policy fixes exact tensor and
operation allowlists, permitted columns, factor/strength ranges, maximum
operations and working elements, targeted-coordinate budget, stored-change
count and fraction budgets, maximum absolute numerical change, and inspection
limits. Unknown fields and unsupported operations fail closed.
The working-element limit bounds both each operation and the union of columns
composed for a single tensor.

Changed-coordinate budgets count **stored-bit differences**, including changes
between positive and negative zero. The denominator for the fraction is all
stored tensor coordinates, including buffers and separately stored aliases.
Numerical differences are reported separately: signed-zero changes consume
the stored-bit budget but are numerically equal, while an edit rounded back to
the original stored value consumes no final changed-coordinate budget. Its
declared coordinates still consume the targeted-coordinate budget. The
maximum absolute numerical change is checked against the saved values.

An accepted candidate is a new bundle containing `model/`, its proposal,
policy, and audit. Edits are prepared in a temporary sibling directory;
existing outputs are refused. Saving does not establish success by itself.
The verifier reopens the files, checks shape, dtype, tensor set and shard
layout, hashes unchanged tensors and assets, counts actual stored and
numerical differences, and enforces the policy against the saved checkpoint.
Only after these checks pass are model and audit published together. Rejected
operations, write failures, verification failures, and audit-write failures
must leave no accepted candidate at the requested destination.

The original must remain stable while it is used. The adapter verifies its
identity again; it never writes the original checkpoint. This API does not
replace operating-system permissions or the later editor/evaluator isolation.

## Bounded inspection and exposure records

Inspection uses exact declared tensors and slices, never executable Python:

```json
{"op": "metadata", "tensors": ["model.language_model.layers.15.mlp.down_proj.weight"]}
```

```json
{
  "op": "statistics",
  "tensor": "model.language_model.layers.15.mlp.down_proj.weight",
  "slices": [[0, 4], [0, 1]]
}
```

```json
{
  "op": "slice",
  "tensor": "model.language_model.layers.15.mlp.down_proj.weight",
  "slices": [[0, 4], [0, 1]]
}
```

Each slice axis is `[start, stop)` with no implicit axis, negative indexing,
stride, or arbitrary gather. Metadata exposes stored names, dtypes, shapes,
shards and identity information. Statistics summarize only the declared
slice, not the whole tensor; slices return its stored numerical values.
Per-request and cumulative limits cap requests, returned values, and scanned
coordinates. A metadata result counts one exposed unit per named tensor;
statistics count six summary values; a value slice counts each returned
coordinate. Both statistics and value slices charge all coordinates in their
declared slice to the scan budget. An append-only exposure log binds the original and policy and
records the requests and returned information before it is released. Reusing
the log carries forward its budgets; starting another CLI process does not
reset that log's accounting. Its hash is retained in candidate evidence.

```bash
python -m weight_space_lab inspect-checkpoint \
  --model artifacts/p50c-acquisition/model \
  --policy configs/candidate_policy.json --request inspection.json \
  --log artifacts/editor-exposures.jsonl

python -m weight_space_lab edit-checkpoint \
  --model artifacts/p50c-acquisition/model \
  --policy configs/candidate_policy.json --patch proposal.json \
  --log artifacts/editor-exposures.jsonl --out artifacts/candidate-001

python -m weight_space_lab verify-candidate \
  --model artifacts/p50c-acquisition/model \
  --policy configs/candidate_policy.json --candidate artifacts/candidate-001
```

The host must control the policy, original, and log destination. An editor
with unrestricted filesystem or Python access could read weights or data
outside this interface. Exposure logging describes this API's releases; it
does not establish process-level isolation.

The Python `propose` and `verify_candidate` methods return full audits for the
trusted host. Audit and inventory files are host evidence, not editor-visible
weight tools. The corresponding CLI commands print only an acceptance or
verification receipt with checkpoint and audit hashes. `inspect` is the
bounded weight-information interface; callers must not expose the full host
inventory or audit to an editor as an unlogged shortcut.

## Deliberately labeled engineering demonstration

```bash
python -m weight_space_lab demo-candidate-edit \
  --model artifacts/p50c-acquisition/model --out artifacts/phase06-engineering

# Explicitly enable the tiny development inference check:
python -m weight_space_lab demo-candidate-edit \
  --model artifacts/p50c-acquisition/model --out artifacts/phase06-engineering-native \
  --backend transformers
```

The fixed hand-authored edit scales column zero of
`model.language_model.layers.15.mlp.down_proj.weight` by 0.99. The demonstration
creates an identity candidate and the edited candidate independently from the
original, records three bounded inspections, verifies both saved bundles,
and checks that an out-of-policy proposal leaves no candidate. It saves the
authored development input `Compute 7 + 5.` and never reads validation or final
benchmark data. Injected write-failure checks are fixture tests, not destructive
experiments on the real original.

The optional native check uses the already pinned Transformers CPU backend,
greedy decoding, seed 17, a four-token output cap, and one request each for the
identity and edited checkpoint. Each arm has a killable worker, 120-second
startup limit, 60-second generation limit, and no retry. Native outputs and
resource metadata are saved separately from scientific run records. There is
no grading, performance claim, feedback-based selection, or Astra call. The
report explicitly labels the intervention `hand_authored_engineering_edit`.
The [real development evidence](../reports/candidate-editing/README.md)
contains measured preservation and saved-change counts, exposure records,
and the deliberately capped native outputs.

## Difference from an exploratory Python kernel

An exploratory kernel can execute arbitrary code, read arbitrary files, load
whole models, derive new statistics, run optimization, change architecture,
and compose unrecorded state. This interface accepts only schema-checked
metadata/statistics/slice requests and the two declared algebraic operations.
It provides no Python evaluation, imports, filesystem paths inside proposals,
network requests, model forward passes, optimizer, autograd, arbitrary tensor
assignment, or unbounded matrix calculation. The separate development smoke
is host-controlled and its output is not automatically fed into an editor.
These restrictions make the supported path auditable, but also make it
strictly less expressive than a general weight-space exploration environment.
