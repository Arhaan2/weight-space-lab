# 128-example IFEval on-device subset — continuation preparation

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2). Upstream: [Google IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) and the preserved WebLLM/MLC browser pipeline.

**Prepared, not authorized or executed.** The supervisor race is repaired in a separately versioned continuation runner. Exactly the 34 unexecuted candidate prompts, original positions 95–128 and 45 instructions, are prepared. All 128 completed baseline results and 94 completed candidate results, original grades, frozen artifacts, and historical ledgers are preserved. The [original incomplete report](../ifeval-browser-v1-heldout/README.md) is unchanged.

The old supervisor could see the next intent after reading an event snapshot that omitted the already durable predecessor completion. The repair rereads and validates durable events immediately before checking sequentiality and reserving/acknowledging the next request. Missing or conflicting predecessors still stop the run. Offline tests reproduce the race and exercise the actual 35-dispatch loop with a fake driver, including late runtime failure and malformed final evidence. No model calls were used for verification.

| Offline verification | Result |
| --- | ---: |
| Python tests: repair, continuation, preparation, merge and existing accounting/analysis | 71 passed |
| Focused browser tests | 17 passed |
| Browser typecheck, driver syntax and build | Passed |
| Independent original-evidence checks | 57 passed |
| Historical files directly checked unchanged by preparation | 858 |
| Prepared manifest file hashes verified | 1,117 |
| Independent final preparation audit | 20 checks passed |

The old independent verifier additionally revalidated its own original historical preservation map and actual source/deployed model bytes. The new browser preparation matches its recorded source/build hashes; generation code, tokenizer, model library, runtime and model assets are unchanged. Public reports contain no private prompts, outputs, authorization text or machine paths.

## Exact additional authorization required

Approve continuation plan SHA-256:

`f366b08d4d9a25d54307af6638b1c5fa66f8ee9b87174e087de18d1ad8f9dad8`

Approval must explicitly permit one new candidate-only load, one warm-up, execution and grading of only the remaining suffix, and a separate combined report under these limits:

| New allowance | Limit |
| --- | ---: |
| Engine loads | 1 |
| Warm-ups | 1, at most 16 tokens |
| Scored requests | 34, at most 1,024 tokens each |
| Generation dispatches including warm-up | 35 |
| Generated tokens including warm-up | 34,832 |
| Generation request deadline | 120 seconds |
| New load-through-cleanup session deadline | 90 minutes |

Both originally authorized engine-load and warm-up slots were consumed and the stopped ledger is closed. Approval therefore creates a new dedicated ledger; no old ledger or clock is reopened, reset or borrowed from. The original 180-minute absolute clock remains immutable. The new 90-minute session clock is separate, with interruption/calendar gap and cumulative active-session time reported independently.

If all remaining requests complete, cumulative totals become three loads, three warm-ups, 256 scored requests and 259 dispatches. Maximum cumulative reservations become 262,192 tokens (16 more than the original ceiling because of the extra warm-up). Prior known generated usage plus the new allowance is at most 148,687 tokens.

Zero retries, baseline reruns, completed-candidate reruns, new proposals, conversions, imports, development reruns or experimental frontier requests. Any runtime/integrity failure or limit exhaustion stops the continuation and preserves partial evidence. Existing candidate cleanup was forced and did not confirm graceful unload; process closure was verified, and live PID/start identities will be checked again before loading.

Execution remains disabled until a separate approval receipt binds this exact plan hash. As prepared: **zero new engine loads, generation dispatches, generated tokens or grader calls; no production ledger or authorization receipt.** Unit tests use temporary fixture ledgers and fake processes only.

## Preservation and future analysis

The frozen merge plan reuses original records and grades verbatim, validates the suffix under its distinct run/configuration identity, and joins only disjoint projections. Only newly attempted suffix outputs will be graded. Before any future completion claim, independently audit the continuation authorization, ledger, events, usage and cleanup; the merge does not replace that execution audit. The final analysis retains all 128 prompts and 173 instructions, the existing strict/loose endpoints, prompt-cluster bootstrap, instruction-count aggregation and missing/error policy.

A future combined report must disclose two candidate sessions, the extra warm-up, the supervisor interruption and any unresolved outcomes. It must preserve the original incomplete report and cannot claim success under the original uninterrupted protocol. The subset remains exposed; editor isolation is unverified and recorded proposal-session provenance is preserved. This is not the full 541-example IFEval evaluation. No new compliance or resource improvement is claimed by this preparation.

Original evaluation freeze: `89ed9703869ab7c8b1032b3d8c5fa59fefed0215d9d0341016684d60b39b8a27`.

Candidate source: `f9c767c06ec3c17dbdfedc43b695d64d025ae7393d3f425cee71ef45c358d6bf`.

Candidate deployed artifact: `6e18cc4906ce18786b9197bcc7fb5f93ffc706d808b0809150349546eef3627b`.

Continuation manifest: `0faa9ec2b548c5070106701fbe965640b44a678c1ee6b185c3d25aa7fcd4ab83`.

See the [machine-readable verification](verification.json) and [execution procedure](../../docs/ifeval-supervisor-continuation.md). The private plan, remaining schedule, provenance map, test log and source/build pins are retained in the separate continuation artifact destination.
