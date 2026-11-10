# Browser V2 preparation freeze

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

The manual exploratory Astra / Ultra study is implemented and frozen. No
experimental editor turn, candidate edit, conversion, package, model load or
target generation was performed. Production execution requires separate later
authorization. V1 remains closed with its evidence and artifacts preserved.

The [protocol](../../docs/ifeval-browser-v2-protocol.md) has byte SHA-256
`d2c72be02ac3bd2f8f2db12cb9b4a705f9837310b117f80fd5a34d71dadf7ec4`.
The seven-day window is November 8, 2026 at 6:32:18 p.m. through November 15 at
6:32:18 p.m. Pacific Standard Time (02:32:18 UTC on November 9–16).
The [configuration](../../configs/ifeval_browser_v2.json) and private preparation
manifest bind the dataset, implementation, saved baseline, runtime and grading.
All freeze and packet digests are in [verification.json](verification.json).

The final set contains **128 prompts and 160 instructions**: 96 singles and 32
pairs, across 18 types and eight families. All 128 constructive responses, 160
targeted strict-failure fixtures and 32 joint-constraint checks passed the actual
pinned official checkers. Browser-tokenizer input counts agree exactly; maximum
rendered input is 88 tokens and maximum constructive output is 51 tokens.
The canonical final manifest digest is
`2ab69feb0c2f12c2fd372da0dfd8d994cb1ed92e82a736d850c8ef9eda7a7bd2`;
its file SHA-256 is
`bd42157a6b052c692addca17cec11645a3e7309fb8c3c16d6fda744a5358abb2`.

This is an AI-assisted custom IFEval-style on-device set using official checkers,
not an official Google IFEval split. Its 26 reused templates and procedural
custody are disclosed in the [sanitized dataset summary](final-set-summary.json).
Deduplication checked 65,425 inventory records with no exact or normalized
prompt/task match. This does not prove semantic uniqueness or absence of
pretraining exposure. Final prompts, answers and fixtures remain private.

The fixed maximum schedule is:

| Allocation | Development | Final | Total |
| --- | ---: | ---: | ---: |
| Sessions | 8 × 32 prompts | 3 × 128 prompts | 11 |
| Scored requests | 256 | 384 | 640 |
| Dispatches including warmups | 264 | 387 | 651 |
| Reserved generated tokens | 262,272 | 393,264 | 655,536 |

Final arms are pristine baseline, selected Astra and selected random control.
Duplicate artifact identities share a final output set. There are at most eight
imports, conversions and packages, four hours local active time including three
browser hours, 32 GiB retained V2 artifacts, and zero automatic retries. The
protocol fixes stage allocations, selection and tie rules, quantization-group
matching, duplicate handling, cleanup and clock semantics. The two editor hours
are an operator planning allowance; unavailable provider metadata remains null.

The first private packet is
`artifacts/ifeval-browser-v2/editor/round-1/editor-packet.txt`.
Its independent [privacy audit](packet-privacy-audit.json) checked 544 final-data
strings with zero matches and confirmed that all 32 baseline feedback rows,
metrics, policies and weight statistics match the pinned evidence. The
[runbook](../../docs/ifeval-browser-v2-runbook.md) contains exact round-by-round
commands, manual turn accounting, failure paths and final-arm selection rules.

**126 focused tests passed**, covering the complete synthetic four-round
workflow, controls, feedback, selection, aliases, restart, dispatch uniqueness,
budgets, cleanup and the preserved V1 components. See [test output](focused-tests.txt)
and [offline session checks](offline-session-validation.json). Preservation checks
found no changes to 3,443 V1 evidence files or 66 pre-existing working files;
the pristine and V1 candidate source/deployment byte identities were also checked.

No concrete implementation defect remains identified by these checks. A real V2
conversion/browser run has deliberately not been tested. Editor isolation remains
unverified, and this is one adaptive trajectory with four conditional matched
controls, not four independent replications.

Qwen, WebLLM/MLC, Google Research IFEval, Ai2 RLVR-IFeval, BrowserLLM, and Gufran
attribution remain preserved.
