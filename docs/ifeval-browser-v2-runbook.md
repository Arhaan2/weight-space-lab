# Frozen browser V2 manual runbook

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

Preparation, the final-set freeze, and first-packet export are complete. No
experimental editor turn, weight edit, conversion, engine load, or target
generation is part of preparation. V1 remains closed. The commands below are
for the later execution stage and require its separate explicit authorization.
Do not use the preparation request as that authorization.

Run from the repository root, in the pinned evaluator environment:

```bash
export PYTHONPYCACHEPREFIX=/tmp/weight-space-v2-pycache
v2_python=artifacts/ifeval-browser-v1/evaluator-env/bin/python
v2_root=artifacts/ifeval-browser-v2
v2() { "$v2_python" -m weight_space_lab.ifeval_v2_study "$@" --root "$v2_root"; }
v2 status
```

The first private packet is
`artifacts/ifeval-browser-v2/editor/round-1/editor-packet.txt`. Never send the
evaluator folder, this authoring chat, a control report, the study state, or a
final result to the experimental editor. Open a fresh authenticated manual
Astra / Ultra session and use only the packet. Selection is operator-enforced;
immutable provider identity, observed effort, frontier usage and marginal
billing remain unavailable. There is no API-key step or nested frontier call.

Once later execution is explicitly authorized, save its original text privately
as `artifacts/ifeval-browser-v2/later-execution-authorization.txt`. Bind the exact
prepared study identity, read without exposing evaluator contents:

```bash
v2_study_sha=$("$v2_python" -c 'import json; print(json.load(open("artifacts/ifeval-browser-v2/ready.json"))["study_sha256"])')
v2 authorize --authorization "$v2_root/later-execution-authorization.txt" \
  --approved-study-sha256 "$v2_study_sha"
```

Before each manual submission, `begin-turn` consumes that round's only editor
opportunity. Paste the corresponding packet once, then save the response
unchanged at the indicated private path. If the turn fails or is abandoned,
use `close-turn`; never submit a replacement. The seven-day deadline is anchored
to the original protocol freeze and is not renewed by any command.

Round 1, after recording the turn and receiving its response:

```bash
v2 begin-turn --round 1
# Manual Astra / Ultra turn; save the original response as response.json here.
v2 import --round 1 --response "$v2_root/editor/round-1/response.json"
v2 build --round 1 --method astra
v2 build --round 1 --method control
v2 evaluate --round 1 --method astra
v2 evaluate --round 1 --method control
v2 export --round 2
```

Round 2 uses the exported second packet and evaluates control before Astra:

```bash
v2 begin-turn --round 2
# Manual turn; save editor/round-2/response.json unchanged.
v2 import --round 2 --response "$v2_root/editor/round-2/response.json"
v2 build --round 2 --method astra
v2 build --round 2 --method control
v2 evaluate --round 2 --method control
v2 evaluate --round 2 --method astra
v2 export --round 3
```

Round 3 evaluates Astra before control:

```bash
v2 begin-turn --round 3
# Manual turn; save editor/round-3/response.json unchanged.
v2 import --round 3 --response "$v2_root/editor/round-3/response.json"
v2 build --round 3 --method astra
v2 build --round 3 --method control
v2 evaluate --round 3 --method astra
v2 evaluate --round 3 --method control
v2 export --round 4
```

Round 4 evaluates control before Astra, then freezes all selected final identities:

```bash
v2 begin-turn --round 4
# Manual turn; save editor/round-4/response.json unchanged.
v2 import --round 4 --response "$v2_root/editor/round-4/response.json"
v2 build --round 4 --method astra
v2 build --round 4 --method control
v2 evaluate --round 4 --method control
v2 evaluate --round 4 --method astra
v2 select
```

These blocks show the successful path. Inspect every command's returned status
before proceeding. Invalid imports close both round slots; skip their build and
evaluation commands. A failed build closes that slot; skip its evaluation.
Failures never refund an attempt. Other eligible slots may continue only when
the state is not halted and cleanup is confirmed. The next packet is unavailable
until both slots of the preceding round close. It includes only permitted Astra
development feedback. Cross-method duplicate grades remain internal and are
withheld from editor feedback.

For an abandoned manual turn, replace `1` with its recorded round:

```bash
v2 close-turn --round 1 --reason operator_abandoned_without_replacement
```

Run final evaluation only after `select` writes the frozen identity plan. Its
`distinct_evaluations` list is authoritative. Execute the following commands in
order **only for arms present in that list**; omitted arms reuse their canonical
artifact's final records and their unused capacity expires:

```bash
v2 final --arm pristine_baseline
v2 final --arm selected_astra
v2 final --arm selected_random_control
v2 analyze
```

If both selections fall back to pristine, run only `pristine_baseline` and
`analyze`. If both select the same non-pristine artifact, run pristine and
`selected_astra`, then `analyze`. No final outcome is returned to an editor and
no proposal is allowed after selection. The existing official grading and
paired prompt-cluster bootstrap are used; the selected-control comparison is
secondary and exploratory.

On restart, inspect durable state first:

```bash
v2 status
v2 recover
```

An interrupted browser session is never resumed. An unresolved stage retains its
reservation and cannot be retried. Recovery without verified process closure
halts the study. Do not delete ledgers, failed attempts, selected artifacts or V1
evidence to unlock capacity. Resume only the next unspent stage when the state
allows it. Hard budget exhaustion and unconfirmed cleanup stop execution.

The maximum schedule is eight 32-prompt development sessions and three distinct
128-prompt final sessions: eight imports/conversions/packages, eleven loads and
warmups, 640 scored requests, 651 dispatches and 655,536 reserved generated
tokens. Local active elapsed time is capped at four hours, including at most
three browser hours; V2 retention is capped at 32 GiB. Exact nontransferable stage
allocations and clock semantics are in the [frozen protocol](ifeval-browser-v2-protocol.md)
and [configuration](../configs/ifeval_browser_v2.json).
