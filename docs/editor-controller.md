# Astra editor interface and development probe controller

Phase 7 implements an offline integration interface, an enforced local editor
process boundary, and a persistent reservation ledger. **It has not executed an
Astra interaction, a search, or a scientific experiment.** The provider-independent
contract is usable while the live connection is pending. The only simulator is
an explicitly labeled Python fixture; it makes no model calls.

The [interface audit](editor-interface-audit.md) records the actually inspected
interfaces. `gpt-6-astra` is an exposed selector in the local catalog and session
tool schema. Codex CLI `0.163.0` is the inspected interface version. An immutable
model version, live returned model identity, API endpoint, entitlement, and
billed usage were not established. None is invented. The controller rejects
live dispatch; neither an imported model-name string nor a fresh chat enables it.

## Access boundary and trusted roles

The operator and broker prepare an approved checkpoint, intervention policy,
and at most 32 development examples. They are trusted to classify that data:
relabeling a held-out question as development cannot be detected from its text.
Do not put held-out data into approved messages, instructions, runtime files,
or policy. The evaluator keeps final tasks, answers, generations, grades, audit
details, and replay results in its host storage. Those files are never mounted
into the editor environment or included in its requests.

The experimental editor role receives only JSON and these six operations:

| Operation | Enforced scope |
| --- | --- |
| `model_info` | Frozen checkpoint identity, architecture names, count, tensor allowlist. |
| `development` | Explicitly approved development messages. No validation or final split. |
| `inspect` | Phase 6 allowlisted metadata, slices, or statistics with persistent exposure accounting. Metadata is projected to shapes, dtypes, and counts. |
| `propose` | Phase 6 bounded patch from the immutable original; independently verify saved bytes and unchanged architecture/tokenizer assets. |
| `probe` | Approved development item IDs, accepted candidate ID or `base`, frozen seeds, backend, decoding, tokens, and time. Only forward generation feedback; no grading or final-evaluation entrypoint. |
| `stop` | Persist session cancellation. |

There is no editor shell, filesystem, network, Python/code execution, training,
gradient, optimizer, architecture, evaluation, or policy mutation tool. Unknown
operations, extra probe controls, paths in candidate IDs, wrong request hashes,
and unsupported proposal fields fail closed. Responses are data, never code.
Trusted checkpoint and inference workers implement fixed handlers; they are
not model-visible tools for arbitrary filesystem or program execution.

The local editor fixture runs under macOS Seatbelt with default-deny permissions
under the same host UID; it does not create a separate Unix account.
It can read its private scratch directory, Python standard library, and narrowly
listed OS runtime locations. It cannot read the repository, evaluator storage,
the broker ledger, host credentials, site-packages, or arbitrary home directories.
It receives a clean environment and no inherited host descriptors. Network is
denied. Process creation is denied by policy and a hard process-count limit.
A separate guardian monitors a parent-owned pipe and a monotonic deadline;
controller death, cancellation, time limit, or output overflow kills the worker.
The scratch directory is discarded. Do not store study assets inside the trusted
OS/Python runtime allowlist. This boundary protects against editor access; it is
not protection against a malicious host administrator.

Unsupported hosts and an outer sandbox that refuses nested Seatbelt profiles
fail closed. There is no unrestricted fallback. Linux isolation is pending;
offline contract and ledger work remains usable without launching a worker.
Actual denial verification uses a host-only synthetic canary and listening
socket, never the final benchmark. See the saved
[isolation verification](../reports/editor-integration/isolation-verification.json).

## Durable budget and execution

Each session has one SQLite ledger with transactional reservations, WAL, FULL
synchronization, frozen configuration and implementation hashes, immutable
attempt IDs, and retry lineage. The request file is published only after its
reservation commits. A retry is another fully charged attempt referring to its
failed predecessor. No automatic retry or action replay occurs.

Accounting distinguishes:

- Editor requests and model probe requests.
- Requested, completed, and failed sequences, including every retry.
- Reserved ceilings, measured usage, and conservatively charged generated tokens,
  elapsed seconds, and billed units when available.
- Pending attempts, interruption, cancellation, and observed limit violations.

The editor request remains active through its tool execution. Checkpoint
inspection, validation, and proposals run in supervised processes. Native probes
reserve their complete item-by-seed plan before checkpoint verification or model
startup, then check cancellation and deadlines while waiting for each sequence.
The CPU inference backend uses `requires_grad_(False)` and `torch.inference_mode()`;
it loads local files with remote code disabled. Generated sequences are bounded
by the frozen `max_new_tokens`. Startup, verification, and teardown consume time.
An observed overrun is retained and cancels the run, never clipped away.

Elapsed accounting sums request windows, including the editor request's tool
execution and its separately reserved probe window. Nested time may therefore
be charged twice; this is deliberately conservative, not unique wall-clock time.
Unverified file imports cannot refund a token/time reservation with a claimed
usage value. Their raw claims remain in the response, while trusted observations
remain null. No provider bill is available in this offline implementation;
fixtures/local probes can report known zero provider charges without inventing
an Astra cost. The ledger supports integer billing units with a frozen unit
definition for a future authenticated adapter.

Cancellation is durable and polled during running work. A reopened session
cannot change its limits, checkpoint, policy, development bundle, probe settings,
or implementation. Both monotonic and wall-clock deadlines persist; clock
regression fails closed. Missing or corrupt ledgers do not reinitialize.
`editor-recover` charges unresolved attempts at their full reservation and marks
unresolved sequences failed. It never reruns an action. Interrupted candidate
directories cannot be probed unless the originating request completed and its
acceptance feedback was saved. An explicitly new directory is a new run, not a
way to resume an exhausted experiment; the editor has no ability to create one.

## Offline operator commands

Use an approved original checkpoint from the earlier phases. Paths supplied to
file imports must not traverse symlinks; on macOS use canonical `/private/tmp`
or `/private/var` paths instead of their aliases. These commands create private
local artifacts, not published results:

```sh
python -m weight_space_lab editor-init \
  --root artifacts/editor-integration \
  --model artifacts/p50c-acquisition/model \
  --policy configs/candidate_policy.json \
  --development examples/editor-development.json \
  --limits configs/editor_limits.json \
  --interface-observation configs/editor_interface_observation.json

python -m weight_space_lab editor-check-isolation
python -m weight_space_lab editor-simulate --root artifacts/editor-integration
python -m weight_space_lab editor-status --root artifacts/editor-integration

python -m weight_space_lab editor-export \
  --root artifacts/editor-integration \
  --instruction 'Inspect only the approved development model information.' \
  --max-output-tokens 512 --timeout-seconds 60

python -m weight_space_lab editor-import \
  --root artifacts/editor-integration --response /absolute/response.json

python -m weight_space_lab editor-cancel --root artifacts/editor-integration
python -m weight_space_lab editor-recover --root artifacts/editor-integration
```

Export writes `requests/<request_id>.json`. Only this approved envelope belongs
in a future isolated editor transport; do not copy the session directory or
provide host tools. The file handoff itself does **not** prove isolation of an
external program that reads it. Such responses stay `unverified_file_handoff`
and cannot establish a live experimental interaction.

The response must contain exactly this structure, with the real request ID
and canonical request hash. `canonical_hash` is the repository's canonical JSON
hash implementation. This example requests information and claims no model:

```json
{
  "schema_version": 1,
  "kind": "editor_response",
  "request_id": "<32 lowercase hexadecimal characters>",
  "request_sha256": "<canonical request SHA256>",
  "evidence_kind": "unverified_file_handoff",
  "model_identity": {"provider": null, "model": null, "version": null, "interface": null},
  "action": {"tool": "model_info", "arguments": {}},
  "usage": {"generated_tokens": null, "billed_units": null},
  "scientific_result": false
}
```

Use `--feedback-id <request_id>` on the next export to include a saved, projected
tool result. Use `--retry-of <failed_request_id>` for an explicitly charged retry.
Every response and action intent is saved before execution. Duplicate responses
are rejected. Invalid responses and proposals never grant additional capabilities.

Probes are disabled by default. To enable them, pass `--probe` at initialization
with an exact object containing `backend`, `seeds`, `max_new_tokens`, and
`timeout_seconds`. The backend is either the existing explicit `fixture` schema
(non-scientific) or the pinned `transformers` specification from Phase 3, with
matching token limits. At most eight frozen seeds, 256 new tokens per sequence,
and 600 seconds per probe are supported. No vLLM/model fallback is performed.
Probe arguments contain only `task_ids` and `candidate_id`. No benchmark scores
are computed and no final-test command is reachable from the contract.

## Validation and pending live integration

Run `python -m unittest discover -s tests -p 'test_editor*.py' -v`. The suite
checks strict contracts, denied tools/test splits, invalid proposals, valid
checkpoint edits, concurrent budget exhaustion, actual process crashes,
restart recovery, retries, cancellation, clock regression, and bounded probes.
Sandbox tests explicitly skip when the OS refuses nested isolation; a separate
host run is needed to establish the denial checks on this machine. Passing
fixtures does not establish any model capability.

`python scripts/verify_editor_integration.py --out artifacts/editor-verification`
persists a complete synthetic integration exercise in a new directory. It
requires a host that permits installation of the restrictive sandbox and makes
no model calls. The [verification report](../reports/editor-integration/README.md)
retains the actual test counts and demonstration evidence.

Before enabling a live development-only Astra interaction, implement and verify
a transport that binds the observed selector to returned identity, prohibits
substitution, exposes only these broker capabilities, enforces token/time/cost
ceilings and cancellation, and reports trustworthy usage. Record whichever model
version the interface actually exposes, including unavailable when appropriate.
The currently inspected collaboration interface inherits broad host tools, and
the CLI inspection did not establish all necessary caps and permissions.
Therefore live Astra access remains pending and **no experiment ran**.

The [live-readiness diagnosis](../reports/editor-live-readiness/README.md)
now separates installed CLI behavior, exact failed conditions, local controls,
provider caps, and post-response provenance. Run `editor-live-readiness --root
<existing-session>` to inspect an existing budget read-only, without relaxing
its source freeze. `editor-prepare-connection --out <new-plan.json>` prepares
one zero-tool acknowledgement request and never reserves or dispatches it.
These commands do not enable the unavailable live adapter or alter the
scientific search protocol.
