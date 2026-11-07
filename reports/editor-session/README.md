# Phase 7 authenticated-session integration

The session variant is implemented locally and its complete mechanical path is
verified with synthetic evidence. **The genuine Astra milestone remains pending.**
No experimental session, Astra proposal, native target-model development
generation, broader candidate search or final benchmark was executed.

## Actual blocker

Authentication is not the missing configuration. The exposed desktop
`create_thread` control selects `gpt-6-astra` and effort, but cannot bind a fresh
session to a restricted filesystem/network/tool profile. It immediately dispatches
the initial prompt. Native UI access to Codex was denied by the computer-use
tool. App-server permission fields establish schema support only, not behavior
of this desktop launch. The existing Python sandbox's passed denial tests do
not establish isolation for an Astra chat.

The smallest missing host capability is a fresh authenticated session whose
actual tool and information boundary can be restricted and verified before
the proposal request is exposed. The same session needs tested cancellation,
no retry after an ambiguous outcome, and an armed deadline supervisor. A separate
API key or nested frontier call would not establish this boundary and was not used.

| Failed session condition | File/function | Expected evidence | Available evidence | Classification and smallest next step |
| --- | --- | --- | --- | --- |
| Actual session capability boundary | `session_controller.py:session_readiness`, `_session_transport` | Launch-bound tool registry and denied access to final tasks/answers/results, unrelated files, network and harness/grader writes | Desktop launch lacks permission arguments; native UI access denied; fixture sandbox tests pass only for their Python launch | Unavailable interface capability: expose a host session launch with a restricted capability set, then test that exact launch |
| Cancellation and no ambiguous retry | Same functions | Confirmed termination and zero automatic retries on the actual session transport | Schema fields exist; desktop behavior is unverified; local fake-provider CLI retry test is a different route | Missing live verification: exercise interrupt and transport failure behavior on the selected route before a proposal |
| Deadline supervisor armed before handoff | `_session_transport`, `SessionController.export_request` | Trusted session supervisor active before export, interrupting at cancellation/deadline | Import checks reject late data; local checkpoint/target workers have active monitoring; external desktop session has no registered supervisor | Missing implementation dependent on supported host controls: wire and verify that supervisor; a ledger deadline is insufficient |

`session-readiness.json` records these gates. `strict-readiness.json` is the
unchanged strict automated diagnostic against the old spent offline ledger. Its
full conditions include file/function, expected/current evidence, classification
and next checks. Its hard frontier cap and legacy budget/source-freeze gates
remain intact; they are not silently relaxed by this separate session variant.

## Changes and frozen attempt

- Added the explicit `authenticated_session_v1` contract, CLI mode and separate
  controller. Its closed request contains approved model metadata, validated
  policy and one development item; its sole response action is a weight proposal.
- Reused the checkpoint validator/editor and supervised native forward worker.
  A valid authenticated response would build and verify one checkpoint, then run
  one baseline/candidate comparison. Immutable action/output records prevent
  replay. Invalid proposals cannot start a probe.
- Kept frontier usage separate from target token accounting. Provider frontier
  generation caps are explicitly unavailable in this variant. Model-authored
  usage and identity cannot authenticate receipts or refund reservations.
- Prepared a separate private attempt: one editor request, one proposal, one
  baseline sequence and one candidate sequence; at most 128 target tokens per
  sequence, 256 total; 300-second shared comparison deadline and 900-second whole
  local attempt deadline. Seed 17, matched frozen greedy CPU settings, thinking
  disabled. No retry or search. `attempt.json` records public hashes and scope.
- Preserved the two prior spent ledgers byte-for-byte. Each still records six
  editor requests, one probe, seven requested sequences and 35 charged tokens.
  `preserved-ledgers.json` records before/after hashes. The new ledger remains
  entirely unspent. The attempted export exited with `SessionBoundaryUnavailable`
  before reservation, publication or dispatch.

The protocol explicitly allows an unavailable immutable model version and
declares the absent frontier provider cap. It does not change final-data
isolation, prohibit fewer scientific operations, or modify the strict protocol.
Training, gradients, optimizer updates, architecture/evaluation changes and
harness/grader changes remain prohibited. No benchmark grader runs in the
development comparison; private feedback records outputs and execution status.

## Model provenance

| Field | Observed evidence |
| --- | --- |
| Requested selector | `gpt-6-astra` |
| Exposed selector | `gpt-6-astra`, desktop tool declaration |
| Requested effort | `high`; explicitly exposed as a supported value |
| Selector accepted by live execution | Unavailable; no launch |
| Observed session/response model and effort | Unavailable; no launch |
| Immutable model version | Unavailable |
| Interface software | Locally inspected Codex CLI `0.163.0`; not a model version |
| Frontier tokens / billed usage | Unavailable; no experimental generation |

No model self-description, invented snapshot, opaque catalog hash or substitute
model fills any unavailable field. Detailed provenance and surface inspection
are in `configs/session_interface_observation.json` and
`docs/session-boundary-audit.md`.

## Verification

The complete regression suite passed **390 tests with two expected skips** in
the outer execution sandbox. The host editor subset passed **65 tests, no skips**.
The new session coverage is **34 tests** within the regression suite, including
the synthetic proposal → validated checkpoint → baseline/candidate feedback path,
closed context, invalid responses/proposals, no substitution, budget exhaustion,
cancellation and conservative restart handling. Simulated receipts and target
outputs remain non-scientific.

The two regression skips map to passing host checks:

| Regression skip | Host result |
| --- | --- |
| `EditorControllerTests.test_isolated_simulator_round_trip_is_non_scientific` | Same test passed |
| `ActualSandboxTests.setUpClass` | All seven methods passed: filesystem/network denials, approved JSON handoff, timeout, cancellation, bounded output flood, invalid responses and controller-SIGKILL cleanup |

These skips reflect nested Seatbelt restrictions and do not explain the live
session block. The actual Astra-session denial test has **not** run. Passing a
synthetic transport test does not change that fact.

Development instructions, prompts, responses, raw feedback, actual session
manifests and machine paths remain under ignored private artifacts. Public
files contain only implementation, synthetic tests, declared controls, hashes
and sanitized status. No credentials were copied or published.
