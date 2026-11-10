# One manual exploratory handoff

The `manual_exploratory_v1` extension of `authenticated_session_v1` accepts one
externally supplied Astra proposal. Its scope is **exploratory workflow
validation, editor isolation unverified**. It is not a controlled research run
or evidence of benchmark improvement. The runner does not launch, authenticate,
supervise or cancel the operator's frontier session. Strict automated/session
readiness remains failed and unchanged.

The explicit amendment preserves the existing session manifest, prepared
request, ledger identity and all accounting history. It records the new source
freeze, approved bounded weight statistics, manual request/packet hashes and a
fixed 48-hour submission expiry. Historical requested effort `high` remains in
the original manifest; the amendment explicitly selects **Astra / Ultra**, with
requested selector `gpt-6-astra` and effort `ultra`. Operator selection is not
runtime metadata. Actual accepted selector, model, effort, immutable version
and frontier usage remain unavailable.

Prepare once against the existing unspent attempt:

```bash
.venv/bin/python -m weight_space_lab editor-session-prepare \
  --root artifacts/astra-session-development/attempt --manual-exploratory
```

The complete packet is private at
`artifacts/astra-session-development/attempt/manual/editor-packet.txt`.
Submit it in the operator's fresh Astra/Ultra session and save the returned JSON
unchanged to `artifacts/astra-session-development/response.json`. No generated
code is executed: the response can contain only a permitted data proposal.
The packet is immutable; neither status nor export renews its expiry.

Run these commands consecutively after receiving the response:

```bash
.venv/bin/python -m weight_space_lab editor-session-import \
  --root artifacts/astra-session-development/attempt --manual-exploratory \
  --response artifacts/astra-session-development/response.json &&
.venv/bin/python -m weight_space_lab editor-session-compare \
  --root artifacts/astra-session-development/attempt --manual-exploratory
```

Import reserves the existing single editor slot before parsing or local work.
Its ledger dispatch event means **local external-proposal ingestion**, not a
frontier dispatch. Human waiting is outside local runtime accounting. A missing
file is not accepted; an existing malformed, invalid or expired submission
consumes the slot. Duplicate submissions, changed request bindings and repeat
comparisons are rejected. There is no automatic retry or replacement candidate.

Import validates the envelope and permitted proposal operations, then uses the
existing supervised checkpoint editor. It reports the actual saved candidate
hash and changed stored-entry count, including zero. The separate comparison
uses the existing forward-only worker and frozen probe specification: one
baseline and one candidate sequence, at most 128 target tokens each (256 total),
with a shared 300-second comparison deadline. The 900-second local deadline
starts at import and is not restarted by the comparison command. Both probe
reservations precede either target dispatch. Failed or interrupted work retains
conservative accounting and cannot authorize another candidate or comparison.

Submission expiry only rejects a late input file. It does not cancel remote
generation or cap frontier reasoning/output tokens. Local cancellation applies
to supervised local work only. Provider-enforced frontier caps remain unavailable;
model-authored token/billing estimates cannot fill runtime usage fields.

Private `manual/import-result.json` and `manual/result.json` record validation
outcomes, execution failures, candidate hash/change count, baseline/candidate
outputs, truncation, and budget consumption. The comparison's raw evidence is
under `manual/comparison/`. None of these files or the packet belongs in public
artifacts. Public status must retain the exploratory label and unverified
isolation. Training, gradients, optimizer updates, architecture/evaluation
changes, harness/grader edits and held-out inputs remain forbidden.
