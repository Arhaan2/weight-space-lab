# Fourth Astra proposal: development packet contract

This is a new proposal after fresh evaluation of pristine, V1 and all three
previous Astra artifacts. The five closed historical studies and their budgets
are not reopened. The new experiment's protocol and ledger govern execution.

The primary objective is strict IFEval prompt compliance above both the freshly
measured pristine score and the best freshly measured existing edit. The
secondary objective is no more sampled output tokens and equal or lower measured
browser resource use. A strict-score gain with more tokens remains a reported
quality/cost tradeoff; resources are measured regardless of quality eligibility.
Tokens, activity rates and allocation descriptors are not interchangeable with
cumulative computation or energy.

The edit family remains fixed: one `scale_columns` operation on each of the
layer 15 and layer 19 MLP down-projection matrices, 60 contiguous columns per
matrix, independent starts from 0 through 3524, and independent multipliers from
0.5, 0.75, 1.25 and 1.5. The source is always pristine. This targets 122,880
coordinates without changing architecture, tokenizer, quantization or decoding.
The earlier column positions and equal factors are not required. A copied prior
proposal does not count as a new intervention.

`scripts/build_fresh_astra_packet.py` exports only after all five 32-prompt
development evaluations complete with resolved grades and verified cleanup. It
checks record/grade identities, configuration and study bindings, and recomputes
scores. The packet contains every development prompt and all 160 corresponding
observations. Identical response strings may reference another artifact within
the same item; different responses and regressions are retained. The approved
weight information consists of the existing fixed spatial statistics, not
task-conditioned activations or semantic feature labels. Historical resource
feedback is explicitly identified as a different cohort. Earlier proposal JSON
did not record rationales, so no retrospective rationale is invented.

The editor receives a fresh context requested as `gpt-6-astra`. Only the indexed
packet parts may be read, with no extra files, network, target inference,
gradients, training, inspection or changes to the evaluation. The parts are
consecutive slices of the exact packet, each at most 12,000 UTF-8 bytes. Their
ordered hashes and exact concatenation are checked before dispatch and import.
All parts must be delivered before the single proposal; packet delivery does not
supply new scientific feedback or create another proposal slot.

The returned JSON uses exactly `schema_version`, `request_id`, `request_sha256`
and `proposal`. The proposal retains the existing bounded schema and explicit
column lists. The host applies and independently verifies saved source changes,
converts with the pinned toolchain, and audits effective quantized changes before
evaluation. Provider-version, effort and usage metadata remain unavailable unless
the actual interface exposes them; the requested selector is not such metadata.

A separate `rationale.txt` may contain a concise decision summary of at most
150 words: the chosen intervention and predicted quality/resource effects, with
uncertainty. The host hashes it alongside the unchanged JSON response before
building or evaluating the new candidate. It is published as a pre-evaluation
prediction, not a measured result, and is not retrospectively revised.

All 32 tasks were already exposed for development. A new measured improvement is
development evidence, not untouched confirmation or a general benchmark claim.
No historical held-out prompt or response is included in this editor packet.
