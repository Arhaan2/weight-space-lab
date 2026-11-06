# Phase 7 control and skipped-test map

This audit describes the local Phase 7 implementation inspected on 2026-11-06.
It distinguishes an offline broker control from a provider-side limit. The
saved host checks establish the Python fixture boundary; they do not establish
that a live Codex/Astra session uses that boundary. No model or benchmark was
executed for this audit.

## The two skipped regression entries

The original [regression log](../reports/editor-integration/regression-tests.txt)
reports `349 tests` and `skipped=2`. These are two **skip entries**, not two
individual sandbox methods:

| Regression entry | Why skipped | Corresponding passing host evidence |
| --- | --- | --- |
| Line 131: `EditorControllerTests.test_isolated_simulator_round_trip_is_non_scientific` | `SandboxUnavailable`: the enclosing Codex sandbox refused nested Seatbelt installation. The test also verifies that the failed attempt stays charged. | Same test at line 31 of [phase7-tests.txt](../reports/editor-integration/phase7-tests.txt). |
| Line 141: `ActualSandboxTests.setUpClass` | The same initial nested sandbox launch failed, so unittest skipped the entire class. | All seven class methods at lines 41–47 of the host log, listed below. |

The seven host methods are:

1. `test_actual_denials_on_host_canary_and_active_network_listener` — file
   contents, metadata, traversal, symlinks, writes, inherited descriptors,
   subprocesses, environment and network denial.
2. `test_approved_json_handoff_works_without_project_imports` — approved JSON
   succeeds and importing the project package fails.
3. `test_cancellation_terminates_running_worker`.
4. `test_controller_sigkill_cannot_leave_worker_running`.
5. `test_invalid_response_objects_are_rejected`.
6. `test_output_flood_is_bounded_and_terminated`.
7. `test_wall_timeout_is_enforced_without_worker_cooperation`.

The [host suite summary](../reports/editor-integration/test-summary.json)
records 58 tests, zero failures, and zero skips. These checks resolve the local
nested-sandbox verification gap. They do **not** resolve live dispatch: at the
start of the audit, `EditorController.dispatch_live` unconditionally raised an
error and launched no executable. Therefore the skipped entries cannot be the
cause of that unconditional gate.

## Exactly what enforces each limit

The default [offline limits](../configs/editor_limits.json) allow eight editor
requests, four probe requests, twelve requested sequences, 4096 charged output
tokens, and 600 charged elapsed seconds. They contain no provider billing cap
(`billed_units: null`). They are limits for the offline session, not proof of
frontier-provider entitlement or remote enforcement.

| Quantity | Enforcement location and behavior | What the evidence does not establish |
| --- | --- | --- |
| Editor requests | [`EditorController._export`](../weight_space_lab/editor_controller.py) calls `BudgetLedger.reserve(kind="editor", sequences=1)` before publishing a request file. [`BudgetLedger.reserve`](../weight_space_lab/editor_budget.py) checks aggregate reserved/charged counts in a SQLite write transaction and refuses overspend. Duplicate IDs and invalid retry lineage are refused. | Publishing a file is not a live provider request; the count is a broker dispatch-intent count. No remote cap is implied. |
| Candidate proposals | [`validate_response`](../weight_space_lab/editor_contract.py) permits exactly one action per accepted response. `_accept` rejects replay of terminal requests and `_execute_action` writes an immutable intent. Thus at most one proposal action can execute per editor request. [`checkpoint._proposal`](../weight_space_lab/checkpoint.py) bounds operations, tensors, columns, numerical parameters, work and coordinate changes. `CheckpointEditor.propose` verifies saved checkpoint assets before acceptance. | There is no independent `candidate_proposals` ledger counter or separate proposal-count limit. Current proposal count is bounded indirectly by editor request count. A connection check requiring zero proposals must refuse proposal actions altogether. |
| Target-model forward probe requests | `EditorController._probe` reserves one `kind="probe"` attempt for the complete approved-item-by-frozen-seed plan before verification/model startup. `BudgetLedger.reserve` checks `model_probes` and `requested_sequences`. `validate_response` rejects unknown development IDs, duplicate IDs, candidate paths and extra controls. | This controls the local target-model service, not Astra's own inference. |
| Target-model output tokens | `_probe_spec` permits 1–256 tokens per sequence and requires the frozen Transformers backend's `sampling.max_new_tokens` to match. `_probe` reserves the plan's token ceiling before launch. [`TransformersBackend.generate`](../weight_space_lab/native_backend.py) passes that value into `GenerationConfig(max_new_tokens=...)`; the actual local generation implementation stops by this bound. `_probe` also rejects returned counts above the bound. | The returned-count check is post-run verification, not the mechanism that stops generation. The local target's `max_new_tokens` says nothing about frontier reasoning/output tokens. |
| Local elapsed time | `BudgetLedger.check_active` checks persistent cancellation plus wall and monotonic deadlines. [`SandboxRunner.run`](../weight_space_lab/editor_sandbox.py) supervises the fixture with an independent monotonic-deadline/lifeline guardian. [`run_checkpoint_action`](../weight_space_lab/editor_tools.py) applies the same guardian to fixed trusted checkpoint handlers and checks active state before candidate publication. `_probe.receive` polls both editor and probe deadlines; [`_Process.close`](../weight_space_lab/native_runner.py) terminates/kills the local process group. Native workers also watch parent death. | This bounds local work. Killing a local frontend or dropping its connection does not prove a provider has stopped generating or billing. Teardown may extend observed time past the deadline; that overrun is recorded and cancels the run. |
| Aggregate charged time | `reserve` refuses aggregate elapsed reservations exceeding the frozen budget. `_finish` reconciles measured time or keeps the full reserved charge when unknown. The parent editor request remains active while a nested probe executes, and both windows are charged. | This is conservative accounting of overlapping request windows, not an exact distinct wall-clock duration and not a remote provider time limit. |
| Frontier output/reasoning tokens | The offline request's `limits.max_output_tokens` is validated, recorded and reserved. No frontier adapter consumes it in a provider request. Simulators report zero model tokens; unverified file imports cannot refund the reserved ceiling. | No provider-side hard generation cap is enforced by this field, ledger reservation, or post-run token usage. An actual supported provider cap, including the relevant reasoning/output semantics, must be demonstrated separately. |
| Frontier provider usage/cost | `BudgetLedger` can reserve a conservative integer billing bound and record observations. Unknown amounts remain unknown; recovery retains bounds. Current offline fixtures/local probes have known zero provider charges, and no Astra billing measurement was obtained. | A ledger's billing reservation does not configure a provider quota. With `billed_units: null`, there is no configured billing ceiling. Observed provider token events would be metering evidence, not a hard billed-usage cap. |

`BudgetLedger.recover_interrupted` consumes unresolved reservations and fails
their unresolved sequences. It does not replay a request. Retry lineage is
explicit and charged again. This is suitable for preserving ambiguity; a future
connection adapter must not automatically retry a frontier request after a
timeout, lost stream, or uncertain remote outcome.

## Launch path and information channels

The tested launch path is:

```text
EditorController.simulate
  -> _export / durable reservation
  -> SandboxRunner.run(_SIMULATOR, approved JSON)
  -> /usr/bin/sandbox-exec + isolated Python standard-library worker
  -> strict response validation
  -> fixed broker action handler
```

`editor_sandbox._profile` denies access by default. The worker has private
scratch, narrowly allowed OS/Python runtime reads, no repository mount,
no site-packages, no network, and no child-process permission. Its sanitized
environment and closed host descriptors are covered by the denial checks.
The final-test canary is a synthetic host-only file; the test does not read or
import actual held-out questions, answers or results. Trusted approved inputs
still require correct operator classification: a development label alone cannot
detect a held-out question deliberately copied into the approved envelope.

The actual frontier launch path at the start of this diagnosis is absent:
`EditorController.dispatch_live` raises before any launcher, provider, CLI,
reservation or session is invoked. `simulate` does not execute Codex or Astra.
`editor-import` validates untrusted offline JSON but cannot control the program
that produced it; it never authenticates that program's identity or isolation.

Consequently, a successful Python denial probe cannot be transferred to a new
CLI launch merely because the CLI accepts a sandbox setting. A live adapter
would need evidence that the actual executable, every model-visible built-in
tool, any dynamic tool, inherited descriptor, plugin/MCP capability, repository
read path and network path use the intended restrictions. Allowing the required
provider transport must not create general network access to unrelated data.
The current Seatbelt profile denies all network, which is correct for the
fixture but does not itself implement a remote provider transport.

The broker's checkpoint and target-model workers are trusted fixed handlers,
not an arbitrary code-execution tool exposed to the editor. Their broader host
access cannot be offered to a live editor as a generic command/file tool.

## Minimal connection-test distinction

The offline contract currently requires the complete six-tool list and permits
`propose` and `probe`. A test promising **no weight changes, no candidate search
and no target-model probes** must use a separate narrowed connection contract
or an enforced action subset; it must not pass an arbitrary live response to
`_execute_action` unchanged. A structured acknowledgement with zero available
tools is enough to test connectivity and response provenance once mandatory
transport controls are demonstrated. It does not require a checkpoint load,
target generation, benchmark schedule, grader, or candidate policy execution.

The generic `fresh_editor_search` protocol is a pilot/full search profile with
development/validation schedules and required `pins.editor_model_version`
([`protocol.validate_protocol`](../weight_space_lab/protocol.py), required pin table
and editor requirements). It is not a Phase 7 one-request connection check.
If an immutable model version is not exposed, an explicit proposed correction
is to record requested selector, interface-accepted selector, observed session/
response metadata, and `immutable_model_version: null` with a documented
availability status for a connection-only check. Do not satisfy the existing
search pin with CLI version, an opaque catalog hash, or model self-description.
Changing the scientific search protocol would require an explicit separate
decision about reproducibility; this audit does not make that change.
