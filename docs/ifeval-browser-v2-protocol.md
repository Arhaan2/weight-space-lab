# IFEval browser V2: proposed bounded adaptive search

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

**Proposal only; no V2 execution is authorized or performed by this closeout.** V1 is complete with inconclusive held-out compliance. This protocol proposes four adaptive Astra candidates and four matched random controls, followed by one frozen evaluation of at most three distinct artifacts on 128 newly supplied prompts. It reuses the existing browser, conversion, grading and durable accounting implementation. It does not reopen V1 ledgers.

## Objective and unchanged measurement

The primary objective is **strict prompt compliance at the fixed browser configuration**: one fresh conversation per prompt, WebLLM 0.2.86, q4f16_1, context 4,096, output allowance 1,024, thinking disabled, temperature 0, top-p 1, seed 42 reset per request, unchanged tokenizer/template/extraction, and the pinned Google checker and package versions. Keep all four compliance metrics and all scheduled prompt/instruction denominators. Record scored sampled tokens, natural completions, truncations, latency, loading, cleanup, and the existing scoped resource measurements as secondary observations. Neither token use nor latency is a selection criterion; no composite efficiency score is proposed.

Reuse the **saved 32-item development baseline**: 16/32 strict prompt, 17/32 loose prompt, 13,941 scored tokens and 263.545 seconds of scored generation. Its content/order identity is `9aa732f4c0dbe06248a93c329299f3fb732a5f148447d3be5a057c93e5fae627`; source checkpoint `f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d`; deployed baseline `5ead3e9c60107ab7168bee5373f6cd636c839b82557fed6550aaaa30170aa5b0`. Verify the complete model, prompt, runtime, extraction and scorer bindings before reuse. No new development baseline dispatch is budgeted. If any binding differs, stop and revise the protocol before search rather than silently spending a baseline rerun. The supervisor repair changes dispatch bookkeeping, not the frozen model or scoring contract.

V1's development improvement was a repetitive formatting-check success. Its 128 completed test prompts are exposed historical evidence, not V2 evaluation data. V2 does not rerun or select candidates on those 128 prompts. Historical V1 findings may inform the search rationale; they cannot supply a new confirmatory result.

## Four rounds, each starting from pristine

For rounds 1–4, request exactly one Astra proposal, import it once, audit its saved source changes, convert once, audit quantization and package once, then evaluate all 32 fixed development items. After each completed round, return its raw development outputs, per-constraint strict/loose outcomes, four aggregate metrics, completion status, token/timing observations, and a brief cumulative table of earlier Astra rounds to Astra **before the next proposal**. Include failed-attempt diagnostics when a round fails, bundled into the next scheduled proposal turn; feedback never creates an extra critique or frontier turn. The baseline packet and existing read-only weight statistics may be reused; no additional target probes or weight-inspection requests are budgeted. Do not provide the new final set or control-arm development outcomes to Astra during search.

Each proposal is applied to a newly verified copy of the same pristine BF16 checkpoint, never to a previous candidate. For this bounded comparison, restrict the existing `scale_columns` operation to exactly one contiguous 60-column block in each of `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight`, both `[1024, 3584]`. Astra chooses each block's start in 0–3524 and its scale from `{0.50, 0.75, 1.25, 1.50}`. This targets exactly 122,880 source coordinates in two operations, within the existing 131,072-coordinate limits. Retain BF16 saved-byte validation, numerical-change limits, pristine-origin checking, and quantization spillover auditing. The proposed factors are inside the existing [0, 2] scale range; the saved-source validator must still verify the actual maximum absolute change ≤1.0, targeted/changed-coordinate limits and all other policy bounds. A failure spends the slot without a weaker replacement edit. Source coordinates, stored changes, decoded changes and affected quantization groups remain distinct quantities.

This deliberately narrow search can test whether feedback guides support selection and scale choices in this family. It does not test arbitrary weight editing or a larger search space. A repeated proposal consumes its slot; reuse its existing identical-artifact development grades, without another model dispatch, and record that fewer unique candidates were evaluated. No extra candidate replaces an invalid, duplicate or quantization-erased proposal.

## Matched random comparison

Each accepted Astra proposal defines one paired random control. The control uses the **same two tensors, 60 columns per tensor, and exact per-tensor scale factors**. For each tensor, uniformly select a different valid block start with the same start modulo 32 as Astra's block. This preserves the size and alignment of the affected 32-coordinate quantization-group footprint as well as exact source support size. It does not match source L2 change or the number of decoded values changed; measure and disclose those differences.

Freeze the random algorithm before round 1: rank eligible alternative starts by SHA-256 of canonical JSON `{seed:20261108, round:r, tensor:t, start:s}` and take the lowest digest, breaking the astronomically unlikely digest tie by the integer start. This is a deterministic pseudorandom ordering independent of development outcomes. Record every eligible start and selected support in the private audit. Do not redraw after import, conversion, grading or disappointing results. An invalid Astra proposal spends its own slot and the associated control slot (the matching quantities are undefined); other technical failures consume the attempted stages without replacement. A valid paired control may still run after an Astra execution failure if cleanup is confirmed and its matching specification is already frozen.

Give controls the same one-import, one-conversion and one 32-item development-session allowance, in the same runtime and scoring configuration. Alternate within-round evaluation order: Astra then random in odd rounds, random then Astra in even rounds. Astra receives only its own completed feedback after both slots close. After round 4, choose the best control by the same development selection rule as the best Astra candidate; generating controls never consults their development outcomes.

The comparison supports a **budget-matched exploratory comparison of adaptive Astra-selected supports with random supports conditional on Astra's per-round factors and edit family**. The controls inherit those scale choices, and one four-round trajectory cannot establish Astra's superiority to random search generally, isolate the causal effect of feedback alone, or identify why a column matters. Report all four matched development differences, each method's selected result, and attrition. Unequal completed-candidate counts after failures weaken the comparison and must remain visible.

## Unused-data audit and proposed final set

The saved V1 preparation manifest accounts for 14,973 Ai2 practice rows: 10,410 eligible, 4,563 excluded/quarantined, and 32 selected development rows. The 10,378 eligible unselected rows are an inventory remainder, not a certified unexposed pool. Ai2 lacks original Tulu task IDs; exact normalized/derived-task comparisons cannot rule out semantic duplicates or pretraining exposure.

The pinned Google source contains 541 prompts: 344 V1-eligible and 197 excluded. Subtracting V1's 128 completed prompts leaves **216 eligible unselected prompts**. An offline ID-only join against the preserved earlier partitions gives:

| Earlier role | All Google items | Overlap with V1's 128 | V1-eligible items outside those 128 |
| --- | ---: | ---: | ---: |
| Development | 377 | 91 | 153 |
| Validation | 98 | 19 | 37 |
| Final | 66 | 18 | 26 |
| Total | 541 | 128 | 216 |

At least one eligible remainder item, Google ID `1000`, already has a saved native development-smoke generation. The older full-source replay is reported as frozen but unexecuted; scheduling alone is not generation evidence. More importantly, historical editor filesystem access was not verified, so absence of a dispatch does not establish absence of exposure. The 26 eligible items in the older final partition outside V1 are insufficient for the proposed 128-item set and lack a complete editor-access attestation. No remaining local pool is certified untouched by this audit.

**Proposed final evaluation:** an independent curator supplies 128 newly authored, previously unpublished tasks, outside the current workspace and any Astra session, with official-checker-compatible constraints from the same 18 supported types and eight families. Use 96 one-instruction and 32 two-instruction prompts, giving exactly **160 instructions per arm**. Each supported type must occur at least four times. The curator fixes the item IDs, prompts, instruction arguments, order, licenses and hashes before the first proposal, without candidate outputs or model-assisted task generation. Apply the existing task-only feasibility rules, exact tokenizer/context check and checker known-answer validation offline. Reject infeasible drafts before the final 128 are sealed; do not substitute items after search starts.

The curator checks exact/normalized prompts and available underlying tasks against all saved V1 and earlier development/validation/evaluation material, and manually reviews close semantic matches. Keep the frozen set in evaluator-only custody inaccessible to the editor's filesystem and tools; give the editor only its size, type coverage and protocol digest. Before search, record the curator's provenance and access declaration and the custody method. This would be an **independent IFEval-checker-compatible set**, not the official Google IFEval benchmark. New authorship and custody support experiment-level non-exposure; they do not prove absence of similar pretraining content.

No such set or custody declaration is supplied in the saved evidence. **This is the principal missing input, and search must not start without it.** If the only available editor route can read that store, move the final material to a separate evaluator system; do not label a merely ignored local directory isolated.

## Selection, final analysis and stopping

Within each search method, select only technically complete, independently graded candidates. Maximize strict prompt passes out of 32. Ties retain pristine when tied with baseline; otherwise prefer the earliest valid round. A method that never strictly exceeds 16/32 selects pristine. Tokens, latency, loose metrics and qualitative judgments cannot break a tie. Freeze the two selected identities before opening the final set. V1's candidate is not an extra selectable arm.

Evaluate each distinct selected artifact and pristine once on all 128 new prompts, at most three arms. If one or both methods select pristine, or both select the same deployed artifact, reuse that arm's single final output set; unused capacity expires. To keep order explicit and simple, final order is pristine, selected Astra, selected control, omitting duplicates, with verified cleanup between loads. Sequential order limits causal latency comparisons and is disclosed.

The primary final contrast is selected Astra minus pristine strict prompt compliance. Compute the frozen-style 10,000 paired percentile bootstrap resamples, seed 0, clustering by whole prompts. Recompute instruction numerator/denominator ratios per resample. Publish all four metrics, paired improvements/regressions/ties and secondary tokens/latency. The selected-control contrasts are secondary and exploratory; no confirmatory multiple-comparison claim is made. An interval spanning zero does not establish improvement; no equivalence claim is possible without a separately specified margin and design. No significance-based stopping, final-set selection, sample-size extension or proposal after final unblinding is allowed.

Stop after round 4 and the single final evaluation, or earlier at any hard limit. An ordinary failed instruction or truncation remains a scored outcome. Invalid/expired proposal, import failure, conversion/precheck failure, erased edit, load failure, unresolved dispatch, deadline, host resource stop, grader error, or unconfirmed cleanup remains a distinct recorded failure; none refunds a slot. Charge an unknown dispatched token count at its reservation. Preserve all raw records, partial outputs, stderr, failed attempts and ledgers. Require confirmed process closure before another arm; if closure is unresolved, stop the whole experiment. No automatic retry, replacement candidate, resumed session or extra warm-up is budgeted. A supervisor interruption closes that session; report incomplete evidence without silently continuing it. Missing/unresolved slots retain scheduled denominators under the existing failure policy, but do not support a completed-effect claim or selection.

## Exact proposed execution ceilings

These are maxima, not promised consumption. Reused V1 evidence consumes zero new dispatches. No capacity transfers between candidates, methods or stages; duplicate/failure savings expire.

| Allocation | New candidate imports | New conversions / packages | Loads / warm-ups | Scored requests | All generation dispatches | Reserved target tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four Astra development slots | 4 | 4 / 4 | 4 / 4 | 128 | 132 | 131,136 |
| Four matched control slots | 4 | 4 / 4 | 4 / 4 | 128 | 132 | 131,136 |
| Final, up to three distinct arms ×128 | 0 | 0 / 0 | 3 / 3 | 384 | 387 | 393,264 |
| **Maximum total** | **8** | **8 / 8** | **11 / 11** | **640** | **651** | **655,536** |

Each development session reserves `32 × 1024 + 16 = 32,784` tokens; each final session reserves `128 × 1024 + 16 = 131,088`. These target-model reservations include the four injected prefix IDs per dispatch; sampled usage is recorded separately. If V1's prefix behavior is unchanged, the maximum sampled output is `640 × 1020 + 11 × 12 = 652,932` tokens. There are **zero additional target-model probes outside the scheduled development/final requests**, and zero baseline development reruns, retry dispatches, extra final arms or extra warm-ups. The existing ledger may name scheduled generations `model_probes`; those remain included in the 651-dispatch ceiling.

| Time/storage limit | Per attempt | Aggregate ceiling |
| --- | ---: | ---: |
| Proposal imports, including saved-source validation | 900 s | 7,200 s across at most 8 |
| Converter package prechecks | 30 s | 240 s across at most 8 |
| Converter subprocesses | 1,800 s | 14,400 s across at most 8 |
| Packaging and deployment audit | 300 s | 2,400 s across at most 8 |
| Browser pre-engine preparation | 300 s | 3,300 s across at most 11 |
| Load through cleanup | 5,400 s per session | **10,800 s summed across all sessions** |
| Each warm-up or scored generation | 120 s | Within browser clocks and dispatch caps |
| Independent grading | 20 s per scored output | 12,800 s across at most 640 |
| Post-generation analysis/export | 600 s | 600 s |
| **All local pipeline stages combined** | — | **14,400 s active elapsed**, including preparation, import, conversion, packaging, browser, grading and export |
| New retained artifacts | — | **32 GiB**, with at least 8 GiB host free space after a stage's reservation |
| Absolute calendar window | — | **7 days** from protocol freeze; no renewal |

The 4-hour aggregate local limit is stricter than the sum of per-stage maxima: they cannot all be reached in one run. Keep a durable accumulated active clock, independent per-session monotonic deadlines, and the absolute calendar deadline. Check the remaining allowance before a stage starts or a model call is reserved; retain the existing cleanup reserve. Stop rather than borrowing future capacity. Existing 5-second host sampling, missing-resource-state stop, memory-pressure level ≥4 and thermal state ≥2 stops remain; GPUBuffer descriptors and WASM capacity are not a physical-memory quota.

Astra has **four requested proposal turns**, no automatic retries or extra critique turns. Proposed bounds per turn are at most **65,536 input tokens**, **16,384 provider-charged output/reasoning tokens**, and **1,800 seconds**; aggregate bounds are **262,144 input**, **65,536 output/reasoning**, **327,680 total frontier tokens**, **7,200 seconds** of editor waiting, and **US$25** maximum provider cost. Use current provider tokenization/pricing before dispatch and reserve the maximum charge; stop if the remaining dollar limit is insufficient. The combined local active time plus editor waiting ceiling is **21,600 seconds (6 hours)**, within the seven-day calendar window.

These frontier-token, reasoning, cancellation and dollar limits are **proposed requirements, not capabilities demonstrated by V1's manual editor route**. V1 did not expose reliable provider usage, billing or backend identity, and a local import timer does not terminate a remote generation. The current manual workflow can bound submitted turns and response-file size (262,144 bytes), but cannot establish those provider ceilings. A metered route with enforceable caps and observed model/effort/usage metadata is a missing execution input; alternatively the user would need to explicitly revise the proposed frontier resource guarantees before any request. Requested Astra identity must remain separate from observed identity. No current request is made by this document.

## V1 costs supporting this size

The [completed development comparison](../reports/ifeval-browser-v1-candidate-recovery/README.md) measured 13,903 candidate scored tokens, 266.793 seconds of scored generation, and 405.731 seconds load through cleanup for 32 prompts. The candidate import took 17.307 seconds; the successful conversion preparation and packaging took 29.880 and 8.742 seconds. Using that one measured candidate as a planning reference, eight development slots require about **111,224 scored tokens and 3,693.280 seconds (61.6 minutes)** including those measured preparation stages. This is an estimate, not a throughput guarantee.

V1's 128-prompt final arm observations were 60,189–68,243 scored tokens and 1,227.986–1,704.563 generation seconds. Its baseline session took 1,740.213 seconds through cleanup; the candidate's two sessions totaled 1,292.011 active seconds and incurred a separate interruption gap. Budgeting three final sessions at the observed slower baseline session gives another **5,220.639 seconds (87.0 minutes)**. Development plus final execution therefore has an observed-scale planning estimate of approximately **2.5 hours locally**, before editor waits and some grading/export overhead; the 4-hour aggregate cap provides limited margin. Approximate scored-token planning range is 291,791–315,953. These observations justify eight development slots as practical on the measured setup; they do not establish an efficiency benefit of the edit or predict a speedup.

## Existing implementation and concrete work before execution

Reuse `ifeval_data` mapping/eligibility/token-counting, `ifeval_handoff` proposal validation/import, checkpoint audits, the recovered converter-interpreter helper, existing `browser_candidate` packaging/decoded-q4 audit, browser generation/extraction, `ifeval_supervisor_gate` durable predecessor repair, official grading, the existing prompt-cluster analysis and durable ledgers. Keep historical source copies and V1 reports immutable.

Small orchestration changes are still needed: fresh candidate roots and ledgers for eight slots; baseline-evidence binding without a new baseline run; the repaired gate in development as well as final sessions; random-control provenance validation (do **not** mislabel a random edit as an editor proposal); per-round feedback export; and a three-arm final schedule plus aggregate clocks/storage/frontier reservations. The original recovery command is bound to V1's failure record and the original final runner schedules two arms; neither should be bypassed or called with fabricated V1 bindings. Adapt their existing primitives with versioned configuration and offline tests. No new inference engine, converter, grader, primary score or readiness framework is proposed.

Concrete missing inputs before execution are (1) the independently authored and custodied 128-prompt/160-instruction final dataset with provenance, exact hashes and access declaration; (2) an editor route that can enforce the stated frontier caps and expose accepted identity/usage, or an explicit revision of those guarantees; and (3) a separate execution authorization binding the completed orchestration revision, final data freeze and these budgets. This closeout supplies the design, not those inputs. Upstream Qwen, WebLLM/MLC, Google Research IFEval, Ai2 RLVR-IFeval, existing BrowserLLM and Gufran attribution remain preserved.
