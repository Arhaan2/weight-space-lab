# Held-out supervisor repair and candidate continuation

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2). Uses the existing [Google IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) checkers and the preserved WebLLM/MLC browser pipeline.

This work repairs the supervisor and prepares only the 34 unexecuted candidate prompts, original schedule positions 95–128 (45 instructions). It does not authorize execution. The original 128-example IFEval on-device subset evaluation remains an immutable, incomplete historical result.

## Race and repair

The original loop read events, then discovered intent files. A predecessor completion could arrive after that read and before the following intent. The sequencing guard therefore rejected a valid next request using stale state. The saved completion for scored item 94 precedes the intent for item 95. That next intent received no ACK, reservation, or model call.

`weight_space_lab/ifeval_supervisor_gate.py` rereads and reconciles durable events immediately before checking the predecessor and before any new reservation or ACK. It validates the contiguous acknowledged/completed prefixes, request and item identities, completed ledger usage, event integrity, deadlines, and runtime failure signals. A missing predecessor still fails immediately. There are no sleeps or retries in the repair. Partial trailing JSONL lines remain withheld by the existing reader.

`weight_space_lab/ifeval_candidate_continuation.py` uses this helper in its actual supervision loop and at every generation gate. The original frozen runner is retained byte-for-byte for reproducibility. The continuation browser patch adds only the candidate-only 34-item schedule schema; generation, extraction, decoding, model files, and grading remain unchanged.

## Prepared scope and additional approval

The private `artifacts/ifeval-browser-v1-heldout-continuation/continuation-plan.json` binds the exact remaining items, source/build hashes, old evidence, merge policy, and limits. Its canonical SHA-256 is recorded in `ready.json` and the public preparation report. Preparation creates no production ledger, authorization receipt, load grant, or execution directory.

Execution needs explicit later approval naming that exact plan hash and granting:

| New allowance | Limit |
| --- | ---: |
| Candidate engine-load attempts | 1 |
| Application warm-ups | 1, at most 16 tokens |
| Scored requests | 34, original positions 95–128 only |
| Tokens per scored request | 1,024 |
| Generation dispatches including warm-up | 35 |
| Generated tokens including warm-up | 34,832 |
| Time per generation request | 120 seconds |
| New session, load through cleanup | 90 minutes |

Both originally authorized load and warm-up slots were consumed. Their ledgers are closed. Unused token or scored-request capacity cannot authorize another engine session. A completed continuation would bring cumulative held-out totals to three loads, three warm-ups, 256 scored requests, and 259 dispatches. Maximum cumulative reservations would be 262,192 tokens, 16 above the original reservation ceiling; known prior generated usage plus the new allowance is at most 148,687 tokens.

The original 180-minute absolute clock is neither resumed nor reset. Approval must grant the separate 90-minute continuation clock, with interruption/calendar gap and summed active-session time reported separately. Before loading, the supervisor checks that no original owned process survives using PID plus native process-start identity. The original candidate's forced cleanup remains disclosed; graceful unload was not confirmed.

Approval must also cover grading the newly attempted suffix and producing a new combined report. There are zero automatic retries, baseline reruns, completed-candidate reruns, new proposals, conversions, imports, development reruns, or experimental frontier requests. Any runtime/integrity failure or limit exhaustion stops the session, retains partial evidence, and requires verified cleanup. No further capacity is implied.

## Execution procedure after approval

Run from the repository root using the preserved evaluator environment. The read-only check is safe before approval:

```sh
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/wsl-continuation-pycache artifacts/ifeval-browser-v1/evaluator-env/bin/python -m weight_space_lab.ifeval_candidate_continuation check --root artifacts/ifeval-browser-v1-heldout-continuation
```

Only after the user approves the exact prepared plan, save the actual approval text privately and pass its path and the approved hash to `authorize`:

```sh
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/wsl-continuation-pycache artifacts/ifeval-browser-v1/evaluator-env/bin/python -m weight_space_lab.ifeval_candidate_continuation authorize --root artifacts/ifeval-browser-v1-heldout-continuation --authorization APPROVAL_TEXT_FILE --approved-plan-sha256 APPROVED_PLAN_SHA256
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/wsl-continuation-pycache artifacts/ifeval-browser-v1/evaluator-env/bin/python -m weight_space_lab.ifeval_candidate_continuation execute --root artifacts/ifeval-browser-v1-heldout-continuation
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/wsl-continuation-pycache artifacts/ifeval-browser-v1/evaluator-env/bin/python -m weight_space_lab.ifeval_candidate_continuation grade --root artifacts/ifeval-browser-v1-heldout-continuation
```

The authorization command binds the later approval to the immutable plan and creates one new dedicated ledger. It rejects reused historical authorization, existing capacity, or changed files. Execution and grading reject absent authorization before creating execution state. Existing execution or grading intents prevent repeat attempts.

After terminal execution and suffix grading, independently audit the new authorization, load grant, ledger, dispatch events, usage, and cleanup before claiming technical completion. The merge validates scientific evidence and does not replace that execution audit. Use `ifeval_continuation_merge.merge_evidence(original_root, continuation_root, continuation_root / "merge-plan.json")` to compute the joined analysis. Save its output in a new continuation report destination. The merge validates original baseline and candidate-prefix records and grades under their true run/configuration identities, validates the suffix separately, and joins disjoint compact projections. It never rewrites or regrades the preserved 128 baseline and 94 candidate results.

Retain all 128 prompts and 173 instructions in the final denominators. Preserve the frozen strict/loose endpoints, paired prompt-cluster bootstrap, instruction-count aggregation, missing/error policy, and qualitative interpretation rules. Report all session budgets, cleanup, unresolved outcomes, changed checker outcomes, and resource limitations. Disclose the extra warm-up and two candidate sessions. A completed continuation cannot be represented as successful completion of the original uninterrupted protocol. The subset remains exposed, editor isolation remains unverified, and proposal-session provenance remains unchanged.

Private prompts, outputs, approval text, detailed provenance packets, and machine paths stay outside public reports. The public preparation report contains only sanitized scope, identities, checks, and authorization requirements.
