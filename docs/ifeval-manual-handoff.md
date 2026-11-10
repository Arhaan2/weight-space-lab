# IFEval browser development handoff

The `ifeval-browser-v1` packet is a private, exploratory on-device experiment.
Editor isolation remains unverified. The baseline stage stops after measured
development responses and packet export; it does not submit a frontier request,
invent a proposal, run a candidate, or generate on the 128-example IFEval subset.

The private packet is `artifacts/ifeval-browser-v1/handoff/editor-packet.txt`.
It contains all 32 practice prompts and observed raw responses, the exact output
extraction policy, per-constraint strict/loose grades, official aggregates,
completion and token/timing evidence, scoped resource observations, checkpoint
and deployment identities, and the existing bounded edit policy. Detailed
evaluation examples and overlap findings are omitted. Requested Astra/Ultra
selection is separate from unavailable observed model/version/effort/usage and
billing metadata.

Weight information comes from 49 new logged read-only inspection requests:
metadata, 16 complete sampled columns per allowed tensor, and eight 60-column
blocks per allowed tensor. These consume 1,015,808 scanned coordinates and
expose 290 metadata/statistic units under the unchanged inspection limits.
The recorded unique geometric coverage handles overlap between slices.
These are sampled statistics, not full-tensor analysis or activation evidence;
no target forward pass is part of inspection. The allowlisted tensors are
layers 15 and 19's `mlp.down_proj.weight`, both BF16 matrices of shape
`[1024, 3584]`. The packet records the actual samples and their limitations.

Use a fresh authenticated Astra/Ultra session for the manual submission. Save
its returned JSON unchanged to `artifacts/ifeval-browser-v1/response.json`.
The response must have exactly `schema_version`, `request_id`, `request_sha256`,
and `proposal`; the UUID and digest must match the packet. The proposal must
have exactly `schema_version`, `base_sha256`, `provenance`, `composition`, and
`operations`, with `provenance: "editor_proposal"`. It may contain only the
existing allowlisted `scale_columns` and `remove_direction_from_columns`
operations. The packet provides their required fields, bounds, and precision
contract. Every candidate starts from pristine weights.

The packet's seven-day submission window is fixed at export and cannot be
renewed by exporting again. It only limits local ingestion; it neither cancels
the remote session nor supplies provider-enforced frontier limits. The import
ledger reserves and consumes its one submission before parsing. Invalid or
expired input spends that slot; there is no automatic replacement. Import
runs saved-checkpoint validation without target inference.

The following are **subsequent-stage commands**, after Astra supplies the
original proposal and candidate execution is authorized. They are not executed
by baseline preparation or packet export. Run from the repository root; keep
the authorization text private at the indicated path. The candidate-stage
authorization must differ from the baseline request that explicitly excluded
candidate execution.

```bash
export PYTHONPYCACHEPREFIX=artifacts/ifeval-browser-v1/pycache
ifeval_python=artifacts/ifeval-browser-v1/evaluator-env/bin/python
ifeval_root=artifacts/ifeval-browser-v1

"$ifeval_python" -m weight_space_lab.ifeval_handoff import \
  --root "$ifeval_root" --response "$ifeval_root/response.json"

"$ifeval_python" -m weight_space_lab.browser_candidate convert \
  --source-original artifacts/on-device-recovery-v1/model \
  --source-candidate "$ifeval_root/candidate/source" \
  --baseline-manifest artifacts/browser-integration-v1/conversion/artifacts.json \
  --converter-python artifacts/browser-integration-v1/conversion/venv/bin/python \
  --policy "$ifeval_root/weight-information/policy.json" \
  --output "$ifeval_root/candidate/converted"

"$ifeval_python" -m weight_space_lab.browser_candidate package-candidate \
  --source-original artifacts/on-device-recovery-v1/model \
  --source-candidate "$ifeval_root/candidate/source" \
  --baseline-manifest artifacts/browser-integration-v1/conversion/artifacts.json \
  --policy "$ifeval_root/weight-information/policy.json" \
  --converted "$ifeval_root/candidate/converted" \
  --output "$ifeval_root/candidate/deployed"

"$ifeval_python" -m weight_space_lab.ifeval_comparison bind \
  --root "$ifeval_root" \
  --candidate-artifact "$ifeval_root/candidate/deployed/candidate-artifact.json" \
  --authorization "$ifeval_root/candidate-authorization.txt"

"$ifeval_python" -m weight_space_lab.ifeval_comparison run --root "$ifeval_root"
"$ifeval_python" -m weight_space_lab.ifeval_comparison compare --root "$ifeval_root"
```

Conversion reuses the pinned MLC converter and q4f16_1 packaging. It retains
the tokenizer, model library, and deployment configuration, and independently
audits actual quantized and decoded changes, including quantization-group
spillover or an edit erased by quantization. Binding rejects an artifact that
does not match the accepted original proposal and pristine base.

Candidate binding freezes the paired analysis manifest before candidate
generation. It reuses the development item order, per-item seed, warmup prompt,
runtime settings, scoring policy, and grader identity. A new candidate ledger
has one load, one warmup, 32 scored requests, and no retries; it cannot reopen
baseline capacity. Analysis uses the existing denominator-preserving paired
pipeline and official instruction-count aggregation. Complete technical
execution, cleanup, and resolved grading are required for selection. A strictly
higher strict prompt-level score selects the candidate; ties retain pristine.
The resulting development comparison does not execute the held-out subset.

All source and packet artifacts stay private under `artifacts/`. Public reports
contain sanitized measurements and scientific provenance. The project retains
Arhaan2 authorship, upstream licenses and scientific attribution, and Gufran's
contribution.
