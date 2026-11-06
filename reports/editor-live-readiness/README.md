# Phase 7 live-integration diagnosis

The live blocker is **an absent adapter plus unsupported/unverified execution
controls**, not the two regression skips and not a missing model-name string.
The installed CLI accepts the requested selector locally, but the tested
controls do not put a provider-enforced generation ceiling into the actual
request. The tested Python sandbox has never launched a live Astra editor or
proved its full tool boundary. Both existing sessions are exhausted offline
fixtures and cannot be reset or promoted into live authorization.

**No live connection test, candidate search, Phase 8, or benchmark run was
started.** Local CLI capture used a synthetic loopback provider that returned
HTTP 400 or dropped the connection before any inference. It is engineering
interface evidence, not a model interaction.

## Checks run first

[Initial checks](initial-checks.json) retain the initial results:

1. `editor-live-readiness` did not exist; argparse returned exit 2. There was
   no pre-existing detailed live-readiness check to run.
2. `EditorController.dispatch_live` unconditionally raised `ValueError`,
   without reserving budget, selecting an adapter, launching a worker, or
   invoking an execution interface.
3. `check configs/profiles/fresh_editor_search.json --require-ready` returned
   exit 2 with 24 missing requirements. That is a later pilot/search checker,
   not a one-request connection checker. None of its null fields was filled by
   guessing or by copying CLI version into model version.

The new `editor-live-readiness` command and `dispatch_live` now share one
evidence-bearing fail-closed gate. The [readiness report](readiness.json)
records every condition, source file/function, expected evidence, available
evidence, classification, and smallest next check. It returns exit 2, correctly.
Run it without changing or resuming the selected ledger:

```sh
.venv/bin/python -m weight_space_lab editor-live-readiness \
  --root artifacts/editor-verification-final/session
```

## Actual CLI controls and provenance

The [behavior audit](../../docs/phase7-cli-controls.md) and
[structured evidence](../../configs/phase7_cli_control_evidence.json) inspect
the actual installed `codex-cli 0.163.0` binary. Its hash matches the current
readiness inspection. A synthetic loopback provider captured real serialization:

- `--max-output-tokens` is rejected. Strict configuration rejects
  `max_output_tokens`, `model_max_output_tokens`, `max_tokens`, `max_cost`, and
  `max_billed_units`.
- Accepted `token_budget` and `rollout_budget` features do not serialize
  `max_output_tokens` or `max_tokens`. The rollout budget adds model-visible
  input. Neither is evidence of a provider hard cap.
- Explicit zero request/stream retries plus
  `features.unbounded_connection_retries=false` produced exactly one request
  when the synthetic server read it and dropped the connection. This concrete
  local behavior is useful for future adapter wiring; it does not prove remote
  cancellation or solve the missing generation cap.
- Some feature switches alter tool descriptions. A complete executable-tool
  allowlist has not been demonstrated. The lack of a top-level `tools` field
  does not mean no tools: this interface also describes tools in code-mode input.

The sanitized, reproducible audit harness is
[`scripts/verify_codex_controls.py`](../../scripts/verify_codex_controls.py).
Its network-only diagnostic profile is explicitly **not** editor filesystem
isolation. No broad filesystem/network permission was added to the editor.

| Provenance level | Current evidence |
| --- | --- |
| Requested selector | `gpt-6-astra`, copied from the actual exposed interface/catalog. |
| Selector accepted locally | CLI configuration accepted it and serialized that string to a synthetic endpoint. This proves local parsing/serialization only. |
| Selector accepted by live execution/provider interface | Not obtained. No real provider request was sent. |
| Observed live session/response model metadata | Not obtained. No model self-description is used. |
| Immutable model version | Not exposed/verified by the inspected surfaces. CLI `0.163.0` is interface software; catalog `comp_hash: "3000"` remains opaque. No snapshot is invented. |

## Limit enforcement and skipped tests

The [control map](../../docs/phase7-control-map.md) gives exact enforcement
functions for editor requests, the indirect one-proposal-per-request bound,
target-model probes and `GenerationConfig.max_new_tokens`, and local process
deadlines/cancellation. These are distinct from frontier generation and usage.

The ledger checks reservations locally and conservatively accounts interrupted
work. It does not configure a remote provider quota. Killing a local client or
receiving a post-run usage event does not establish a remote hard cap or prove
remote cancellation. The selected budgets have `billed_units: null`: they do
not configure a monetary ceiling. Accordingly the readiness diagnostic does
not invent a mandatory money-cap blocker from that null. Any future configured
billed ceiling needs a defined unit and enforceable bound; available provider
usage must still be recorded, with unknown values preserved.

The original two skips map exactly to passing host checks:

| Original regression skip | Passing host evidence |
| --- | --- |
| `EditorControllerTests.test_isolated_simulator_round_trip_is_non_scientific` | The identical test passed in the saved 58-test host suite. |
| `ActualSandboxTests.setUpClass` | All seven methods in that class passed: actual denial checks, approved JSON handoff, cancellation, controller SIGKILL, invalid response, output flood, and wall timeout. |

The skip cause was refusal of nested Seatbelt installation by the enclosing
Codex sandbox. The successful host tests retain the restrictive worker profile.
They establish the Python fixture boundary, not a live CLI/editor boundary.
The source launch path is `simulate -> SandboxRunner -> sandbox-exec -> Python`.
`dispatch_live` still launches nothing. Therefore the actual editor/tools cannot
currently be asserted to use the tested boundary.

## Final implementation checks

The final [test summary](test-summary.json) records **356 regression tests**
with the same two explicit nested-sandbox skip entries, and **65 Phase 7 host
tests** with no failures or skips. The seven new readiness checks prove that
the command and controller share the gate, preparation cannot dispatch or
reserve, response content cannot carry actions/identity claims, and read-only
budget inspection cannot reset an exhausted offline session. The prerequisite
Phase 5/6 staged tree separately passed all 298 tests before its preservation
commit.

## Prepared connection check

[The prepared plan](connection-test-plan.json) contains one structured request:
return exactly `{"status":"acknowledged"}`. It has an empty tool list, an exact
zero-action response validator, zero candidate proposals, zero target-model
probes/tokens, a proposed 64-token frontier bound including reasoning, and a
30-second local bound. It solicits no self-identification and includes no
checkpoint, weights, held-out data, or benchmark. It has no automatic retry;
an ambiguous remote outcome must remain indeterminate and conservatively charged.

Preparation neither reserves nor dispatches. The separate connection validator
cannot route a response to `_execute_action`; the existing six-tool offline
contract is not reused. The plan's bounds are proposed requirements, **not**
proof that the current CLI can enforce them and **not** new budget authorization.

The selected historical ledger retains six of six editor requests and seven
of seven sequences used, leaving zero requests/sequences for this check. It is
also frozen to the previous implementation. The diagnostic reads it in SQLite
read-only mode, verifies its manifest binding, and preserves every spent charge.
The normal resume freeze remains enforced. No fresh ledger was created to get
around exhaustion, and no code hash was rewritten to make resume succeed.

## Explicit protocol correction proposal — not applied

The scientific `fresh_editor_search` profile currently requires a nonempty
`pins.editor_model_version`. If no immutable build is actually exposed, that
cannot truthfully be filled with CLI version, a catalog hash, an alias, or model
self-description. For a **connection-only diagnostic**, propose a distinct
provenance record containing requested selector, interface acceptance,
session/response metadata, `immutable_model_version: null`, and an explicit
`not_exposed` availability status. Record an actual version if one is returned.

This would establish transport/identity evidence at the granularity the
interface exposes, not bitwise model reproducibility or scientific equivalence
across dates. It does not relax token limits, tool isolation, no substitution,
or budget controls. The existing scientific search protocol and its required
version pin have **not changed**. Any later scientific relaxation requires an
explicit protocol decision and a documented reproducibility limitation.

The generic pilot also requires target checkpoint/runtime/tokenizer, datasets,
seeds, grading, and split-audit pins. Those are appropriate for a search pilot
but unnecessary for this zero-probe connection check. No pilot was initialized
to satisfy a diagnostic.

## Remaining prerequisites

1. A supported execution interface for the requested selector that actually
   enforces a frontier generation bound including reasoning. The tested CLI
   controls do not provide it; a reservation/instruction/timeout is no substitute.
2. A live adapter whose exact executable, every available tool, filesystem and
   network route use a demonstrated editor boundary. Preserve denial of final
   tasks, answers, outputs, grades, and undeclared channels.
3. Binding of the narrow connection contract, no-fallback behavior, tested retry
   settings, local supervision, and explicit remote-outcome semantics.
4. An explicitly authorized, frozen live budget with remaining capacity once
   the capability gates pass. Existing exhausted offline budgets stay intact.
5. Capture actual interface acceptance and response identity/usage during the
   eventual gated request. These are post-response evidence obligations, not a
   circular demand to possess a real response before the first safe request.

The full condition tables follow below. No failed condition is converted to
verified merely because a configuration field exists.

## Detailed live conditions

| Condition; source | Status / category | Expected evidence | Available evidence | Smallest concrete change/check |
| --- | --- | --- | --- | --- |
| `live_adapter_missing` — `weight_space_lab/editor_controller.py::EditorController.dispatch_live` | blocked / missing_implementation; pre_dispatch | Executable adapter selecting the requested model, binding the narrow contract, controls and returned provenance. | Dispatch is fail-closed; simulate() calls a Python fixture only. File imports remain unverified. | Implement one adapter only after the selected interface demonstrates the mandatory controls. |
| `actual_launch_isolation_unverified` — `weight_space_lab/editor_sandbox.py::SandboxRunner.run / _profile` | blocked / missing_live_verification; pre_dispatch | Denial tests on the exact editor launch: final tasks/answers/results, host files, descriptors, plugins and network routes. | Host Python fixture denials passed; CLI executable/provider network are not permitted by this profile. | Build a narrow provider broker and verify the exact launch/tools; do not widen the Python profile or infer equivalence. |
| `frontier_generation_cap_not_demonstrated` — `weight_space_lab/editor_controller.py::_export / dispatch_live` | blocked / unavailable_interface_capability; pre_dispatch | Supported dispatch control observed on the actual provider request and documented/enforced at the provider. | max_output_tokens is an offline envelope/reservation; local deadlines and output usage events are not remote generation caps. | Use an explicitly supported provider-enforced output limit for this exact selector, or propose a protocol change; do not treat CLI goals as a cap. |
| `frontier_billed_cap_not_demonstrated` — `weight_space_lab/editor_budget.py::BudgetLedger.reserve / finish` | not_configured / unavailable_interface_capability; disclosure | If a billed ceiling is configured: establish its unit and enforceable conservative bound before dispatch. Otherwise record available usage without claiming a monetary cap. | {"actual_frontier_billed_usage":null,"configured_billed_units_limit":null,"local_fixture_zero_is_frontier_usage":false} | Do not invent a mandatory money cap from billed_units:null. Any future configured cap requires demonstrated enforcement. |
| `no_substitution_control_unverified` — `configs/editor_interface_observation.json::interfaces / model_selector_sources` | blocked / missing_live_verification; pre_dispatch | Explicit requested selector, no fallback behavior, and execution-interface model/reroute metadata. | gpt-6-astra is exposed; allowProviderModelFallback exists in experimental schema. No real dispatch acceptance or response was observed. | Wire and test no-fallback behavior before launch; reject reroute/mismatch using interface events. |
| `authorized_live_budget_unavailable` — `weight_space_lab/editor_live.py::_budget_snapshot` | blocked / configuration; pre_dispatch | A non-cancelled authorized live ledger with one request/sequence, 64 frontier tokens and 30 seconds available, with complete billing semantics. | Offline session; requests remaining 0; sequences remaining 0; code freeze matches: False. | Preserve spent offline budgets; explicitly authorize/freeze a live budget only after capability gates pass. Never reset a run. |
| `existing_session_code_freeze_changed` — `weight_space_lab/editor_controller.py::EditorController.__init__ / _implementation` | blocked / configuration; pre_dispatch | Exact original package/runtime hashes for any resume, or an explicitly authorized new protocol with budget provenance. | Read-only diagnostic access works; ordinary controller resume correctly refuses changed implementation. | Keep the historical session immutable. Do not relax the source freeze or recreate its ledger to obtain live budget. |
| `zero_action_contract_prepared` — `weight_space_lab/editor_live.py::connection_test_plan / validate_connection_response` | prepared / missing_implementation; pre_dispatch | One structured acknowledgement request, empty tools, exact response validator, no calls to _execute_action. | Separate inert zero-action plan and exact acknowledgement validator are implemented; no transport exists. | Bind this narrow contract in the future verified adapter; do not reuse the six-tool offline action contract. |
| `remote_cancellation_and_retry_semantics_unverified` — `weight_space_lab/editor_live.py::connection_test_plan` | blocked / missing_live_verification; pre_dispatch | One provider request, bounded local supervisor, no retry after an ambiguous remote outcome; explicit remote cancellation capability or limitation. | Local worker cancellation is tested. It cannot prove a remote provider stops generation when a local client is killed. | Verify transport retry/cancel behavior and charge ambiguous outcome conservatively; retain provider hard cap independently. |
| `accepted_selector_not_observed` — `configs/editor_interface_observation.json::actual_returned_model_identity` | not_observed / missing_live_verification; post_response | Interface-produced session/dispatch acceptance for gpt-6-astra, separate from configured request and model text. | null | Record live session/dispatch metadata during the gated one-request check. |
| `response_model_metadata_not_observed` — `configs/editor_interface_observation.json::actual_returned_model_identity` | not_observed / missing_live_verification; post_response | Interface response model/provider/reroute metadata, never the model's self-description. | null | Capture authenticated interface events; reject mismatches and retain unknown fields as null. |
| `frontier_usage_not_observed` — `weight_space_lab/editor_budget.py::BudgetLedger.finish` | not_observed / missing_live_verification; post_response | Record interface token usage and billed usage when exposed; preserve unknowns and charge ambiguous outcomes conservatively. | CLI schema exposes token usage events. Actual provider usage and billing are null, not fixture zero. | Capture metering during the gated connection check; do not describe it as an execution cap. |
| `immutable_model_version_not_exposed` — `weight_space_lab/protocol.py::validate_protocol / _PROFILE_PINS` | not_exposed / unavailable_interface_capability; protocol_proposal | An immutable build value actually supplied by the interface, or an explicitly approved protocol revision. | {"cli_version_is_model_version":false,"immutable_model_version":null,"opaque_catalog_comp_hash":"3000"} | Propose explicit unavailable-version provenance for a connection diagnostic; keep the existing search protocol unchanged pending approval. |

## All 24 legacy search-check failures

Every row is from `weight_space_lab.protocol.validate_protocol` applied to
`configs/profiles/fresh_editor_search.json`. These remain later-pilot
requirements; none is silently promoted into a connection prerequisite.

| Missing field | Category | Expected evidence | Currently available | Smallest next step |
| --- | --- | --- | --- | --- |
| editor_provenance.model_id | missing_live_verification | Actual editor execution-interface provenance, separate from request text and model self-description. | Requested selector and installed CLI evidence exist; no live editor execution metadata exists. | Record interface acceptance and response metadata in the gated connection check before using it in a future pilot. |
| editor_provenance.interface | missing_live_verification | Actual editor execution-interface provenance, separate from request text and model self-description. | Requested selector and installed CLI evidence exist; no live editor execution metadata exists. | Record interface acceptance and response metadata in the gated connection check before using it in a future pilot. |
| tokenizer_revision | configuration | A verified, frozen tokenizer_revision for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| runtime_version | configuration | A verified, frozen runtime_version for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| chat_template_sha256 | configuration | A verified, frozen chat_template_sha256 for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| checkpoint_sha256 | configuration | A verified, frozen checkpoint_sha256 for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| editor_model_version | unavailable_interface_capability | An immutable editor build actually exposed by the execution interface. | No authenticated immutable build is exposed; CLI 0.163.0 and opaque comp_hash 3000 cannot fill this pin. | Obtain a genuine immutable version or approve an explicit provenance-schema correction; do not silently relax the pilot. |
| split_audit_sha256 | configuration | A verified, frozen split_audit_sha256 for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| backend | configuration | A verified, frozen backend for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| precision | configuration | A verified, frozen precision for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| execution_limits.max_sequences | configuration | A verified, frozen execution_limits.max_sequences for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| execution_limits.max_duration_seconds | configuration | A verified, frozen execution_limits.max_duration_seconds for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[0].dataset_revision | configuration | A verified, frozen benchmarks[0].dataset_revision for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[0].task_manifest_sha256 | configuration | A verified, frozen benchmarks[0].task_manifest_sha256 for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[0].grader_revision | configuration | A verified, frozen benchmarks[0].grader_revision for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[0].truncation_policy | configuration | A verified, frozen benchmarks[0].truncation_policy for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[0].item_ids | configuration | A verified, frozen benchmarks[0].item_ids for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[0].seeds | configuration | A verified, frozen benchmarks[0].seeds for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[1].dataset_revision | configuration | A verified, frozen benchmarks[1].dataset_revision for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[1].task_manifest_sha256 | configuration | A verified, frozen benchmarks[1].task_manifest_sha256 for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[1].grader_revision | configuration | A verified, frozen benchmarks[1].grader_revision for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[1].truncation_policy | configuration | A verified, frozen benchmarks[1].truncation_policy for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[1].item_ids | configuration | A verified, frozen benchmarks[1].item_ids for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
| benchmarks[1].seeds | configuration | A verified, frozen benchmarks[1].seeds for the later candidate-search pilot. | The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed. | Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check. |
