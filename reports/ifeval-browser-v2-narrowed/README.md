# Browser quality and resource study: awaiting Round 1

Maintainer: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

The narrowed workflow is implemented, verified and frozen. **No V2 editor submission, candidate import, conversion, model load, warm-up or target generation has occurred.** The first private Astra / Ultra packet is exported; the next dependency is the operator-supplied original response. The existing study authorization is already bound. [V1 remains closed and inconclusive](../ifeval-browser-v1-final/README.md).

Use the [current protocol](../../docs/ifeval-browser-quality-resource-protocol.md) and [canonical command runbook](../../docs/ifeval-browser-v2-runbook.md). The earlier V2 preparation had zero submissions and experimental stages; it remains unchanged in its original private tree, with its original calendar record. Its random controls and authored final set are outside this study.

## Data and selection

The [official-source audit](../../docs/ifeval-v2-final-data.md) froze **0 final prompts / 0 instructions**. Of 541 official prompts, 344 meet existing runtime feasibility rules. V1's 128 final prompts plus one saved native-smoke item are confirmed exposed; historical exposure for the remainder is unresolved. No new generation or expected performance informed this decision. Fresh final-quality confirmation is unavailable; no authored replacement or V1-test reuse is implied.

Development reuses the verified 32-prompt pristine baseline: strict 16/32, loose 17/32, scored output tokens 13,941. A complete valid candidate must improve strict prompt compliance, use no more scored output tokens, and satisfy frozen deployment/memory limits. Rank by strict compliance, fewer tokens, then earlier round. Pristine wins quality ties. If no edit qualifies, retain pristine and skip redundant comparisons.

## Resource integration and capacity

The [adapter derivation](resource-preset-derivation.json) binds the original Browser LLM Bench standard profile at revision `b39187b1aff1f218493622d7fcf3cd5d24773b41`. It preserves 60 measured requests, three windows of 20, two 90-second gaps, original baseline/loaded-idle/recovery intervals and reset behavior, temperature 0, top-p 1, seed 42, thinking disabled, context 4,096, output cap 64, one 16-token warm-up and 60-second request limit.

Once an edited artifact is selected, only pristine A and selected B enter A–B–B–A sessions. The local model registry binds their audited deployed bytes; the historical leaderboard is not the control. Native unprivileged CPU/process-memory/GPU collection uses the original collectors and report merger. Power is optional/unavailable when absent. Allocation descriptors, address space and physical footprint remain distinct; original energy intervals and signal coverage are retained. Two sessions per artifact are a small pilot, not evidence of precise equality.

| Remaining allocation | Count |
| --- | ---: |
| Manual submissions / candidate imports, conversions, packages | 4 / 4 each |
| Development task requests | 128 |
| New final-task requests | 0 |
| Standard resource requests | 240 |
| Loads / warm-ups | 8 / 8 |
| Total dispatches | 376 |
| Reserved target-token ceiling | 146,560 |
| Browser time | 90 minutes: 60 development + 30 resource |
| Active local processing | 150 minutes after unused final allocation expires |
| Retained artifacts | 32 GiB ceiling |
| Automatic retries | 0 |

Consumed experimental capacity is zero. Preparation retains about 456 MiB, including conservative accounting of linked pristine assets; the exact snapshot is in [verification.json](verification.json). The seven-day window runs from **November 10, 2026 at 12:55:06 p.m. through November 17 at 12:55:06 p.m. Pacific Standard Time**. Waiting consumes calendar time, not active processing. The larger authorization ceilings remain recorded, but unused final capacity cannot fund extra work.

## Verification and provenance

The full workspace passed 682 offline tests against the frozen source (two skipped). The public clean export passed 621 offline tests (three optional tests skipped). Final core regressions passed 44 tests; the resource adapter passed eight. Browser typecheck, 88 JavaScript tests, 21 native harness tests and the production build passed. The actual frozen worker/client bundle also built successfully through the pinned dependency path. These checks used synthetic orchestration fixtures, not model evidence. Public V1 closeout verification passed; all 3,656 protected historical evidence files remained unchanged, and unrelated working files were preserved.

Cleanup removed the obsolete initial-repository publisher, consolidated current workflow documentation and reused canonical validation/conversion/supervisor/benchmark helpers. Historical runners, patches, controls, authored-data validators, failed attempts and native replay functionality remain for demonstrated reproduction needs. See the [cleanup record](../../docs/cleanup-quality-resource.md).

Implementation commit: `3222a3a`. Study SHA-256: `bf38a30cb9bd0bac1b4afc2167b9293c32bb4d61b1af3e178a0f474e502b4e0c`. The exact 72-file local source snapshot is immutable and private; it includes pre-existing working modules outside the publication commits. The public commit and full local snapshot identities are distinct, and exact local-state replay requires retained private evidence. Old source hashes were not rewritten.

The first private packet is `artifacts/ifeval-browser-v2-narrowed/editor/round-1/editor-packet.txt`; its request and packet digests are recorded in [verification.json](verification.json). It contains real saved development feedback and approved weight information. The independent [packet audit](packet-privacy-audit.json) passed 92 integrity/privacy checks against actual saved evidence, with no final prompts, authorization or raw host paths in the packet. No final material, private authorization or session transcript is published. Editor isolation is procedural/unverified, and provider model-version/effort/token/billing metadata remain unavailable.

**V2 development results, final task results and standard-resource results are all absent.** No joint-goal success, quality preservation, speedup or resource equality is claimed. With no untouched final set, any later development improvement will still lack fresh final-quality confirmation.

Qwen, WebLLM/MLC, Google Research IFEval, Ai2 RLVR-IFeval, the motivating weight-space research, and Browser LLM Bench authors Gufran Mirza and Arhaan Aggarwal retain attribution.
