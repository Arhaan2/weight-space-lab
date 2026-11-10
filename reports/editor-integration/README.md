# Phase 7 offline integration verification

Verified on **2026-11-06 (America/Los_Angeles)**. The offline editor interface,
development probe controller, persistent budget ledger, and local OS boundary
are implemented. **No Astra interaction, candidate search, or scientific
experiment ran.** All editor responses in this verification are explicitly
synthetic fixtures. No final benchmark asset was opened for the denial checks.

## Execution-interface provenance

The [inspection record](../../configs/editor_interface_observation.json) retains
the actual exposed selector `gpt-6-astra`, display name `GPT-6-Astra`, and
inspected Codex CLI version `0.163.0`, with binary/schema hashes. Its opaque
catalog `comp_hash` is recorded as `3000`, without asserting that it identifies
an immutable model version. Actual live identity, immutable model version,
API endpoint, entitlement, and billed usage remain unverified or unavailable.
The [audit](../../docs/editor-interface-audit.md) explains why the available
collaboration/CLI surfaces were not used to bypass the required controls.

## Verified controls

| Verification | Evidence |
| --- | --- |
| Host file contents, metadata, path traversal, symlinks and writes denied | Actual Seatbelt execution against a synthetic evaluator-only canary. |
| Shell/Python subprocesses, network and inherited descriptors denied | Actual forbidden operations; loopback was listening, so network denial was not inferred from connection refusal. |
| Timeout, cancellation, output flood and controller SIGKILL | Supervised worker termination tests with independent guardian. |
| Reservation, concurrent exhaustion, retry lineage, restart and clock regression | Transactional SQLite ledger tests, including committed state surviving process exit. |
| Invalid proposals, forbidden tools/test splits and fixed probe parameters | Contract, checkpoint-worker and controller tests. |
| Complete offline interaction and restart recovery | Persisted synthetic request/response, inspection, proposal, probe, rejection and recovery exercise. |

[Isolation evidence](isolation-verification.json) records eleven actual denial
checks and source hashes. This uses a process permission boundary under the
same macOS host UID, not a separate Unix account. Trusted OS/Python runtime
files remain readable; repository/evaluator files do not. Unsupported hosts or
blocked nested sandbox installation fail closed.

The [test summary](test-summary.json) and raw logs retain both runs:

- [Full regression suite](regression-tests.txt): **349 tests**, zero failures,
  two explicit nested-sandbox skip entries.
- [Phase 7 host suite](phase7-tests.txt): **58 tests**, zero failures, zero
  skips, including the actual sandbox and isolated simulator interaction.

The host run was necessary because the enclosing Codex sandbox refuses nested
Seatbelt installation. The workers themselves still ran under the restrictive
editor profile. No unrestricted editor fallback was used.

## Persisted integration demonstration

[Integration evidence](integration-verification.json) includes the frozen source
hashes, responses, budget snapshot, and all passing checks. Reproduce with a new
output directory:

```sh
.venv/bin/python scripts/verify_editor_integration.py --out artifacts/editor-verification
```

The demonstration constructs four synthetic F32 weights and a synthetic
development example. It performs isolated simulated model-info and inspection
requests, a bounded proposal, one fixture probe of the accepted candidate, and
an invalid proposal. It then abandons one reserved file-handoff request, reopens
the session, reconciles it conservatively, and verifies budget exhaustion.

The recovered ledger retains six editor requests, one probe, seven requested
sequences, six completed sequences, one failed interrupted sequence, and no
pending requests. The fixture reports three synthetic output tokens; the
interrupted request's full 32-token reservation remains charged, producing a
35-token accounting total. These are integration counters, not measured Astra
usage or scientific output. Known fixture provider charges are zero; no Astra
billing data was obtained. Retry behavior is verified separately in the tests.

## Remaining gate

A real development-only Astra interaction remains pending an authenticated
adapter with enforceable tool/filesystem isolation, no substitution, hard
generation/time/billing limits where available, cancellation, and trustworthy
usage reporting. File imports remain unauthenticated and non-scientific. See
[operator commands and contract](../../docs/editor-controller.md).
