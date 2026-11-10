# 128-example IFEval on-device subset

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

The held-out comparison is incomplete; no completed instruction-compliance gain is established.

Evaluation status: **incomplete**. This is the final evaluation of the frozen selected candidate; no further proposal, development rerun, or test expansion was performed.

**Candidate values below include zero credit for 34 unexecuted prompts (45 instructions). They are incomplete-schedule accounting, not measured failures or a completed estimate of the candidate's effect.**

| Compliance | Pristine | Candidate | Difference (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: | ---: | --- |
| Strict prompt (primary) | 69/128 (53.91%) | 50/128 (39.06%) | -14.84 | [-23.44, -6.25] |
| Loose prompt | 74/128 (57.81%) | 55/128 (42.97%) | -14.84 | [-24.22, -6.25] |
| Strict instruction | 106/173 (61.27%) | 77/173 (44.51%) | -16.76 | [-25.27, -8.38] |
| Loose instruction | 112/173 (64.74%) | 83/173 (47.98%) | -16.76 | [-25.73, -7.82] |

The candidate session stopped after 94 completed scored requests when the supervisor sequencing guard rejected the pending gate for scored item 95. The durable completion of item 94 precedes that next gate, but the supervisor had not reconciled it in its live event snapshot. The preserved logs and frozen control flow support an event-snapshot/intent-file race; exact snapshot-read timing was not instrumented. No out-of-order engine generation was demonstrated. The gate for scored item 95 received no acknowledgement, ledger reservation, or model call, leaving 34 slots unexecuted. This was a supervisor failure, not an observed model failure for those slots or an exhausted execution limit. The candidate session required forced termination: worker/browser/harness process cleanup was confirmed, all 14 known owned process IDs were absent on recheck, and graceful engine unload was not confirmed. No implementation repair, retry, or rerun was applied within this evaluation.

This table retains the frozen zero-credit policy for missing or unresolved slots. The displayed differences and intervals describe that projection; even an interval excluding zero cannot establish an instruction-compliance gain or degradation from this incomplete comparison.

All 128 scheduled prompts and 173 scheduled instructions remain in the denominators. Instruction scores use counts, not an average of prompt percentages. Intervals use 10,000 paired bootstrap resamples, seed 0, with whole prompts as clusters; instruction ratios are recomputed within each resample. The existing analysis has no exact paired significance test. Significance was not a completion condition.

Strict prompt pairs: **8 improvements, 5 regressions, 81 ties**, and 34 unresolved pairs.
Loose prompt pairs: **10 improvements, 5 regressions, 79 ties**, and 34 unresolved pairs.

| Execution observation | Pristine | Candidate |
| --- | ---: | ---: |
| Natural completions | 69 | 55 |
| Truncations | 59 | 39 |
| Unexecuted slots | 0 | 34 |
| Missing outputs (including unexecuted slots) | 0 | 34 |
| Attempted model failures | 0 | 0 |
| Ambiguous dispatch outcomes | 0 | 0 |
| Attempted outputs with unknown token counts | 0 | 0 |
| Grader errors | 0 | 0 |
| Grader timeouts | 0 | 0 |
| Scored sampled output tokens | 68,243 | 45,588 |
| Scored response seconds | 1,704.563 | 968.893 |

Actual aggregate consumption: 2/2 engine loads, 2/2 warm-ups, 222/256 scored requests, 224/258 generation dispatches, and 113,855 known reported generated tokens against the 262,176-token limit. 0 dispatched requests have unknown token counts. Reservations total 227,360 tokens; conservative charged usage is 113,855 tokens. First load through final cleanup: 2,786.134 seconds. Automatic retries: 0; pending requests: 0; recorded limit violations: 0. Zero proposals, development reruns, or experimental frontier requests.

Executed warm-ups used the frozen prompt and at most 16 tokens each, are included in the budget, and are excluded from all task scores. Four injected empty-thinking prefix IDs count toward max_tokens but not sampled completion usage, permitting at most 1,020 sampled tokens for a 1,024-token request and 12 for a 16-token warm-up.

Baseline load through cleanup: 1,740.213 seconds. Cleanup applicable; confirmations: unload_confirmed=True, worker_terminated=True, chrome_exit_confirmed=True, local_harness_exit_confirmed=True, controller_exit_verified=True, chrome_os_exit_verified=True, forced_termination=False, owned_browser_and_worker_resources_released=True; remaining owned processes: 0.
Candidate load through cleanup: 1,022.176 seconds. Cleanup applicable; confirmations: unload_confirmed=False, worker_terminated=True, chrome_exit_confirmed=True, local_harness_exit_confirmed=True, controller_exit_verified=True, chrome_os_exit_verified=True, forced_termination=True, owned_browser_and_worker_resources_released=True; remaining owned processes: 0.

Frozen order: pristine then candidate, one engine at a time; candidate loading required verified pristine unload. Both arms were bound to fresh conversations, thinking disabled, temperature 0, top-p 1, seed 42 reset per request, context 4,096, output cap 1,024, WebLLM 0.2.86 and q4f16_1. Request deadlines remained 120 seconds, per-model deadlines 90 minutes and aggregate deadline 180 minutes.

| Resource observation (sampled maximum) | Pristine | Candidate |
| --- | ---: | ---: |
| Worker GPUBuffer descriptor bytes | 1,747,697,720 | 1,747,697,720 |
| Worker WASM address-space bytes | 283,181,056 | 283,181,056 |
| Worker JS heap bytes | unavailable | unavailable |
| Worker timer delay ms | 15,305.599999904633 | 9,645 |

A single sequential comparison does not demonstrate a small speed or energy improvement. The workloads are unequal: 128 pristine scored requests versus 94 candidate scored requests. Aggregate response times, tokens, and resources therefore cannot establish a speed or energy gain. Descriptor allocations and WASM address space are not physical VRAM. Owned-process memory, CPU time, disk bytes, kernel energy and host-wide numeric ranges are preserved separately in the JSON when available. CPU, disk and kernel-energy values are cumulative counters since each process was created, reported as maximum single observed process counters per role; they are not task-only deltas or sums across processes. Kernel energy is not battery energy. Shared allocations, unrelated host activity and sample spacing limit these observations. No controlled energy improvement was demonstrated.

The sole development improvement (schedule item 17) required at least 22 highlighted sections. The baseline echoed long plot text with 3 highlighted spans; the candidate repeatedly emitted the same short title, producing 169 complete highlighted spans before truncation. Both used 1,020 sampled tokens. The official strict and loose fail-to-pass grades remain unchanged. This gain reflects repetitive content satisfying the formatting checker; it does not demonstrate better substantive answering or reduced repetition.

Only the 94 executed and independently graded pairs are interpreted; 34 candidate slots remain unexecuted and unresolved. Seventeen executed pairs changed at least one strict or loose instruction outcome. Strict prompt outcomes among those executed pairs comprise 8 improvements, 5 regressions, and 81 ties; loose outcomes comprise 10 improvements, 5 regressions, and 79 ties. Changes are mixed: some gains reflect closed JSON, bounded length, required keywords, or reaching a postscript, while several gains are achieved through repetitive content. Some regressions accompany shorter and less repetitive responses because minimum-length checks are lost. These qualitative observations do not change official grades, introduce a new grader, or establish substantive answer correctness. The interrupted 128-prompt comparison does not establish a completed held-out instruction-compliance gain.

- Schedule item 6: strict prompt tied failure; loose prompt tied failure. Both prompt-level outcomes remain failures. The baseline repeats JSON members until truncation and never closes the object. The candidate stops naturally with a short, closed JSON object, gaining the JSON instruction. It still contains the forbidden word and does not provide a substantive riddle. This is a formatting and termination improvement, with the other checked constraint still failing.
- Schedule item 9: strict prompt improved; loose prompt improved. The baseline repeatedly emits the same promotional bullet block until truncation and exceeds the word limit. The candidate stops naturally within the limit, although it still duplicates one bullet block. This is improved adherence to the maximum-length instruction with less repetition in this response; it does not establish broader answer quality.
- Schedule item 12: strict prompt regressed; loose prompt regressed. The baseline reaches the required separator and second response before entering a repetitive loop. The candidate repeats a clause in the first response until truncation and never reaches the separator or second response. Both are repetitive and truncated; the candidate loses the checked two-response structure. The structural grades do not verify factual claims in either response.
- Schedule item 15: strict prompt regressed; loose prompt regressed. The baseline avoids the prohibited word and stops naturally. The candidate repeats commentary that explicitly includes the prohibited word and truncates. The regression is an observable forbidden-word violation within repetitive content, without a separate judgment of translation correctness.
- Schedule item 19: strict prompt regressed; loose prompt regressed. The baseline produces a draft that contains the required keywords. The candidate instead asks for more details and offers to draft later, omitting the checked keywords. Both stop naturally. The regression reflects failure to provide the requested constrained response in the current turn.
- Schedule item 26: strict prompt improved; loose prompt improved. Both responses are long, repetitive, and truncated, and both satisfy the sentence-count requirement. The candidate includes the required keyword set that the baseline misses. This is an observable keyword-compliance gain; the checker does not verify the factual claims or usefulness of the discussion.
- Schedule item 28: strict prompt tied failure; loose prompt improved. The baseline loops in uppercase until truncation and never reaches the required ending. The candidate stops naturally after including that ending. Its mixed-case ending loses the strict uppercase instruction while gaining the ending instruction, so strict prompt compliance remains a failure. The saved loose grading accepts both constraints. This is a tradeoff between checked formatting requirements, not evidence of substantive correctness.
- Schedule item 29: strict prompt improved; loose prompt improved. The baseline repeatedly restarts sections until truncation without the required postscript marker. The candidate includes a marked postscript before the token limit, while still repeating some material and receiving a length stop. The gain reflects reaching an explicitly requested response component; it does not demonstrate a fully coherent or complete plan.
- Schedule item 39: strict prompt tied failure; loose prompt improved. The baseline has extra paragraph breaks and places the required starting word in separate locations. The candidate produces three prose paragraphs and a final paragraph consisting only of that word. The saved strict grade still fails while the loose grade passes. This is a permissive formatting success rather than evidence of a substantive fourth paragraph or reliable content.
- Schedule item 40: strict prompt regressed; loose prompt regressed. The baseline passes the minimum-word checker after substantial repetition, including repeated word-count assertions, and truncates. The candidate stops naturally with less repetitive prose but falls below the required word count despite an approximate word-count claim. The checker regression accompanies shorter output; it does not by itself establish worse substantive writing.
- Schedule item 51: strict prompt regressed; loose prompt regressed. The baseline mentions the required postscript marker in a refusal-like explanation, then repeats placeholder blocks until truncation. That marker mention receives the saved official postscript credit even though it is not a substantive closing postscript. The candidate repeats short placeholder-filled lines until truncation and lacks the marker, retaining placeholder credit but losing postscript credit. The regression illustrates checker permissiveness as well as repetitive output in both arms.
- Schedule item 64: strict prompt tied failure; loose prompt tied failure. Both responses fail the surrounding-quotation instruction. The baseline reaches the minimum sentence count through repeated claims before truncation. The candidate stops naturally with fewer sentences and loses sentence-count credit, while prompt-level failure remains tied. Reduced length and repetition here do not establish factual correctness, and the official grade only reflects the specified checks.
- Schedule item 65: strict prompt improved; loose prompt improved. The baseline repeats prose with a comma and no qualifying highlights until truncation. The candidate repeatedly cycles through a few highlighted lines without commas and also truncates. Both checked constraints improve, but the highlight threshold is met through repeated content rather than sustained development of the requested story. This is a repetitive checker gain similar in character to development item 17.
- Schedule item 73: strict prompt improved; loose prompt improved. The baseline begins a JSON object but repeats text inside a field until truncation, leaving the structure unfinished. The candidate stops naturally with a closed JSON object accepted by the existing checker. This is improved output-format completion. The checker does not establish that the product features or claims are supported.
- Schedule item 80: strict prompt improved; loose prompt improved. The baseline gives a brief calculation with commas. The candidate removes commas by repeating the same calculation in bullet form until truncation. Both add unit prices instead of accounting for the stated quantities, so the punctuation-compliance gain does not repair the arithmetic error. This qualitative observation leaves the official compliance grade unchanged.
- Schedule item 89: strict prompt improved; loose prompt improved. The baseline repeatedly cycles through verses until truncation without the required postscript marker. The candidate stops naturally and includes the marker, while still repeating phrases and lines. This is a checked-component and termination gain, not proof of improved thematic or literary quality.
- Schedule item 94: strict prompt improved; loose prompt improved. The baseline stops naturally before reaching the required sentence count. The candidate elaborates and then repeats the same sentence many times until truncation, satisfying the minimum-sentence checker. The gain is materially supported by repetitive padding and does not demonstrate a better substantive rubric.

The official scores are unchanged by this qualitative inspection. Instruction compliance and substantive answer correctness are separate; no post hoc grader was introduced.

Frozen identities (all four model identities were verified against actual existing bytes before generation):

- Baseline source checkpoint: `f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d`.
- Baseline deployed artifact: `5ead3e9c60107ab7168bee5373f6cd636c839b82557fed6550aaaa30170aa5b0`.
- Candidate source checkpoint: `f9c767c06ec3c17dbdfedc43b695d64d025ae7393d3f425cee71ef45c358d6bf`.
- Candidate deployed artifact: `6e18cc4906ce18786b9197bcc7fb5f93ffc706d808b0809150349546eef3627b`.
- Evaluation freeze: `89ed9703869ab7c8b1032b3d8c5fa59fefed0215d9d0341016684d60b39b8a27`.
- Frozen 128-item content/order: `7493091404188ce790e2d6fd61c8a82c79ef8d9e0d8731015e127396d53284e2`.
- Original complete data manifest: `6d577377506b74cca45b56a8c9245c1ffde99598dd2871bde06475c173b7d676`.
- Original Google source revision: `e6890f85757dd84e27ca6df2dd30651dafad28e0`; source file: `67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`.

The [machine-readable report](result.json) preserves configuration, tokenizer, template, runtime, model-library, grader, paired-analysis and resource identities. The [independent verifier](verification.json) checks evidence, schedule, accounting and preservation. Private prompts, outputs, session packets, authorization records and machine paths remain outside these public artifacts.

Limitations: this is a 128-example subset, not the full 541-example IFEval evaluation. Frozen eligibility and coverage selection do not establish representative full-benchmark performance or absence of pretraining exposure. The bootstrap describes one preselected comparison. This remains exploratory on-device work with editor isolation unverified. The supplied proposal-session provenance reports file-reading and JSON-parsing tools used on the development packet, with no external research, additional weight inspections, or target forward passes; that provenance was not independently observed. The selected weight columns are not a verified causal explanation of response behavior.

The evaluated subset is now exposed. Its feedback was not used for another weight edit, and it cannot later be called an untouched test set for further edits.

Upstream attribution remains with [Google Research IFEval (Zhou et al.)](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval), [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), Qwen, WebLLM/MLC, existing BrowserLLM work, Gufran’s contribution, and their preserved licenses.
