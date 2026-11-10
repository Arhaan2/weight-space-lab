# Authenticated session editor, protocol v1

`authenticated_session_v1` is a separate, explicitly authorized development
attempt using the existing signed-in Astra session route. It does not request an
API key or issue a nested frontier-model call. Astra's intended experimental role
is to propose a bounded weight edit. Repository authorship remains the maintainer's.

The local implementation includes request/response validation, a durable ledger,
candidate checkpoint construction and validation, and a paired development
comparison. A production session transport is **not registered**: the available
desktop launch cannot bind a fresh session to a demonstrated filesystem, network,
and tool boundary. Preparing an attempt does not dispatch it. Synthetic test
receipts exercise the full local path but establish no Astra execution.

## Explicit protocol differences

The strict automated mode and `editor-live-readiness` remain unchanged. This
variant removes the requirement for a provider-enforced frontier generation-token
ceiling, which is unavailable through the audited subscription route. It also
permits an unavailable immutable model version, recorded as null. These are
declared protocol differences, not evidence that either capability exists.
Results from this variant would have a different resource-control guarantee from
the strict automated protocol and must identify the variant.

Requested selector is `gpt-6-astra`, requested effort is `high`. A trusted launch
receipt must establish interface acceptance of that exact selector and prevent
fallback. Observed session/response model metadata, observed effort and immutable
model version are recorded separately when exposed; absent fields stay null.
Conflicting observed model/effort metadata is rejected. CLI software version is
not a model version. Model-authored identity fields must be null. No handoff JSON,
model self-description, operator assertion or configuration flag authenticates a
session or grants access.

## One bounded attempt

`configs/editor_session_protocol.json` and `configs/editor_session_probe.json`
declare the attempt before execution:

| Scope | Declared limit | Enforcement |
| --- | --- | --- |
| Editor requests / candidate proposals | 1 / 1; no retries | `SessionController.export_request`, durable ledger, immutable action intent and response consumption |
| Development comparisons | 1; baseline 1 sequence, candidate 1 sequence | `run_development_comparison`, immutable output and two pre-dispatch reservations |
| Target generation | 128 tokens per sequence, 256 total | Frozen backend `max_new_tokens`, existing native generation stopping, worker token-count validation and ledger |
| Target comparison elapsed time | 300 seconds shared by both arms and verification | Supervised checkpoint/target workers and comparison active checks |
| Whole local attempt | 900 seconds from request export | Persisted reservation clocks; candidate worker checks and callback throughout comparison; late responses rejected |
| Frontier generation tokens | No provider-enforced cap available | No cap claimed; record independently exposed session usage |
| Provider billing | No configured hard cap | Record available trusted receipt usage; unknown remains null |

The ledger's token counter is explicitly **target-model-only** in this variant.
Its three requested sequences are one editor response plus the two target
sequences. Frontier tokens and available billing are recorded in the separate
private receipt and sanitized public summary, never taken from the model's
claims. Unknown remote usage cannot be charged as a fictional bounded maximum.
The 1,500-second ledger elapsed reservation capacity is the sum of one 900-second
editor reservation and two 300-second probe reservations. It is not additional
wall-clock permission: the whole-attempt deadline remains 900 seconds.

The target is the existing immutable checkpoint. Both arms use the same one
privately authored development item and seed 17, CPU bfloat16/eager execution,
four threads, the pinned chat template, greedy decoding and thinking disabled.
These are explicitly frozen development settings; no benchmark configuration or
grader is changed. Feedback includes actual outputs, token counts, truncation
and timing, with no claim that one development pair estimates benchmark benefit.

Local acceptance deadlines and supervised target workers cannot stop a remote
session by themselves. A live transport must provide a tested session monitor,
cancellation and no automatic retry after an ambiguous outcome before export is
enabled. Until then the live-readiness gate remains closed. Interrupted requests
retain reservations; recovery requires confirmed executor termination and never
replays a proposal or comparison. Existing spent ledgers are neither reset nor
promoted into this separately authorized attempt.

## Access boundary and handoff

Only a closed projection of approved model metadata, validated intervention
policy and one development item enters the request. Prior feedback is null.
Allowed response action is one `propose` operation under the existing checkpoint
policy. The trusted broker applies and verifies saved weights, then runs the
already frozen comparison. The editor receives no checkpoint filesystem paths,
Python execution, unrestricted file/network access, harness writes or grader
writes. Training, gradients, optimizer updates, architecture changes and changes
to evaluation remain forbidden by the contract and existing checkpoint/native
implementations.

Those broker restrictions do not prove that a desktop session lacks other tools.
The actual editor launch must also deny final tasks, answers and results; host
files; inherited descriptors/environment; plugins/connectors; unrelated network
access; and writes to the broker, harness, graders and ledgers. A fresh history,
projectless chat, worktree, prompt instruction or test of a different executable
cannot supply that evidence. See `docs/session-boundary-audit.md`.

Private preparation uses the existing CLI with the new mode:

```bash
python -m weight_space_lab editor-session-readiness
python -m weight_space_lab editor-session-init \
  --root artifacts/astra-session-development/attempt \
  --model artifacts/p50c-acquisition/model \
  --policy configs/candidate_policy.json \
  --development artifacts/astra-session-development/development.json \
  --protocol configs/editor_session_protocol.json \
  --probe configs/editor_session_probe.json \
  --interface-observation configs/session_interface_observation.json
python -m weight_space_lab editor-session-prepare \
  --root artifacts/astra-session-development/attempt \
  --instruction-file artifacts/astra-session-development/instruction.json
```

`editor-session-export` checks the trusted transport before reserving or exporting.
It currently fails without dispatch. Once an actual adapter is verified,
`editor-session-import --root ... --response ...` verifies its receipt, validates
the proposal, constructs one checkpoint and performs the one comparison. There
is no arbitrary command/provider/plugin option or JSON safety override. The
existing automated controller rejects session-mode roots and vice versa.

Development prompts, requests, responses, raw outputs, receipts, credentials and
machine-specific paths stay in ignored private artifacts. Public artifacts may
contain protocol/configuration hashes, allowlisted provenance and usage fields,
sanitized validation results and explicit pending status. Never publish a raw
`session.json`, request, response or comparison directory.
