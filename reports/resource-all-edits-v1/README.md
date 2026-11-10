# Resource comparison of every existing edit

All ten scheduled sessions completed: two each for pristine Qwen3.5-0.8B, the V1 manual edit and the three Astra edits. Every session produced 60 natural completions, 2,340 measured output tokens and three warm-up tokens. There were no target failures, truncations or model-call retries. The recorded device, browser and runtime matched across sessions; a documented pause for filesystem recovery interrupted the schedule.

The original joint goal remains unproven. All three Astra edits previously scored 16/32 strict development passes, matching the saved pristine score; none qualified under the frozen quality rule. This separate resource study does not rerun quality or change the pristine selection. The historical quality baseline used Chrome 156 while the edited development runs used Chrome 157; those comparisons do not isolate weight changes alone. The V1 held-out result was separately inconclusive. See the [completed quality report](../ifeval-browser-v2-collaboration-run/README.md) and [V1 final report](../ifeval-browser-v1-final/README.md).

Each row below is the unweighted mean of **two session observations**. Requests within a session are not independent experimental replications. Native footprint is the mean of the two sampled session peaks in decimal GB, not a single physical VRAM measurement. Full-response throughput includes the full response time and is not decode-only throughput.

| Artifact | Mean response, ms | Full-response tokens/s | Mean session peak footprint, GB | CPU capacity, % | GPU command-buffer time, ms/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pristine Qwen3.5-0.8B | 617.85 | 63.12 | 3.092 | 3.700 | 607.86 |
| V1 manual edit | 601.06 | 64.90 | 3.075 | 3.644 | 616.19 |
| Astra round 1 | 652.12 | 60.15 | 2.995 | 3.438 | 588.09 |
| Astra round 2 | 601.70 | 64.82 | 3.054 | 3.570 | 621.26 |
| Astra round 3 | 1055.28 | 42.94 | 2.895 | 2.696 | 420.58 |

The mean response times for V1 and Astra R2 were lower than pristine; R1 and R3 were higher. All four edited mean session peaks were lower than pristine, but the pristine peaks themselves ranged from 2.987 to 3.198 GB. Two observations, a shared host and the interrupted schedule do not establish resource equivalence or a causal efficiency improvement.

R3's two response means were 661.28 and 1,449.28 ms. Both observations remain included. Lower CPU/GPU activity rates alongside slower responses do **not** establish less total computation or energy. CPU is the original interval-weighted share of total logical-CPU capacity during active phases. GPU time is experimental driver-accounted command-buffer time for the owned Chrome tree, including rendering; queues may overlap, and it is not a utilization percentage or model-only measurement. Power and energy were unavailable and remain null.

Sampled GPUBuffer descriptor maxima were identical at 1,747,697,720 bytes, and sampled WASM address-space maxima were identical at 283,181,056 bytes in all ten sessions. These allocation counters are not physical memory and must not be added to native footprint. Scaling weight values preserved tensor shapes, architecture and the deployed quantization/runtime controls.

| Position | Artifact | Mean response, ms | Load, seconds | Sampled peak footprint, GB | CPU coverage | GPU coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Pristine Qwen3.5-0.8B | 620.68 | 2.93 | 2.987 | 89.9% | 92.1% |
| 2 | V1 manual edit | 592.53 | 2.79 | 3.002 | 87.2% | 92.1% |
| 3 | Astra round 1 | 602.85 | 2.87 | 3.010 | 90.9% | 93.1% |
| 4 | Astra round 2 | 598.33 | 2.54 | 3.123 | 91.2% | 93.4% |
| 5 | Astra round 3 | 661.28 | 2.52 | 2.994 | 91.3% | 93.4% |
| 6 | Astra round 3 | 1449.28 | 36.42 | 2.797 | 96.8% | 97.8% |
| 7 | Astra round 2 | 605.06 | 2.40 | 2.986 | 92.4% | 94.7% |
| 8 | Astra round 1 | 701.40 | 2.62 | 2.980 | 88.1% | 96.2% |
| 9 | V1 manual edit | 609.59 | 2.43 | 3.148 | 90.0% | 92.5% |
| 10 | Pristine Qwen3.5-0.8B | 615.02 | 3.33 | 3.198 | 89.6% | 94.4% |

The fixed mirrored order was pristine, V1, R1, R2, R3, R3, R2, R1, V1, pristine. Each artifact's mean schedule position is 5.5; this is neither randomized nor fully counterbalanced. R3's two sessions both preceded the pause, while the other artifact pairs straddled it. Matching recorded hardware/runtime does not establish constant background host conditions.

The original Browser LLM Bench standard profile was retained: 60 measured requests in three windows of 20, two 90-second loaded-idle gaps, one 16-position warm-up, 64-position measured cap, resetChat per request, temperature 0, top-p 1, seed 42, thinking disabled and context 4,096. The original baseline/loaded-idle/recovery intervals and collector/timeout/memory protections remained in force. Every session used a fresh dedicated Chrome profile and empty IndexedDB weights with verified local no-store fetching; OS filesystem caches were not cleared and could be warmed by integrity verification. All five artifacts shared the exact model library, tokenizer and `mlc-chat-config.json` bytes. Runtime was WebLLM 0.2.86 / q4f16_1 on Chrome 157.0.8103.39, Apple M4 Pro, 14 logical CPUs and 24 GiB RAM. See the [frozen protocol](../../docs/resource-all-edits-protocol.md), [configuration](../../configs/resource_all_edits.json) and [research methodology](../../docs/research-methodology.md).

After position 6, an integrity read of a newly offloaded historical input exceeded its bounded 25-second materialization allowance. Position 7 had no session directory, reservation, load or generation. The stopped state and its six observations were preserved, and a separate continuation was frozen for only the unused positions 7–10. One no-load continuation preparation also stopped on an offloaded pristine shard. Exact frozen bytes subsequently materialized normally; pinned NumPy 2.3.5 imports were moved to an exact cached-wheel overlay without changing the installed environment or package version. Independently verified caches retained exact input bytes for any further repair; the completed continuation needed 0 additional cache restorations.

The measured gap after position 6 was 1331.75 seconds (22.20 minutes). The full pause before continuation preparation of 1230.00 seconds was conservatively charged to the original local allowance, including offline recovery and the 54.95-second failed no-load preparation. No completed session was replaced, and no model call was retried. The original total ceilings and 24-hour deadline remained unchanged.

Actual resource-study usage was **10 loads, 10 warm-ups, 600 measured requests, 610 total dispatches and 23,430 sampled output tokens**. The reserved cap was 38,560 output positions. Charged browser time was 3038.87 seconds (50.65 minutes) of 90 minutes; charged local time, including the conservative pause, was 4314.04 seconds (71.90 minutes) of 120 minutes. Conservative retained storage was 9.429 GiB of 32 GiB, counting both execution roots, the failed preparation and exact-byte caches. There were zero frontier calls, quality generations, unresolved attempts or target-sequence failures. One original pre-dispatch verification failure and one offline preparation failure remain recorded. All ten browser/worker/harness/native-capture cleanups were verified.

The [final JSON](result.json) includes every session summary, 610 sanitized request rows, both replicate values, coverage, descriptive deltas, immutable model/settings/build identities and continuation accounting. Request rows contain timings, token counts and finish reasons; prompts and generated text remain private. The independent terminal audit passed 12,229 checks against retained private evidence, including raw request rows, model payloads, bindings, arithmetic, cleanup and original-state preservation.

Public provenance is available in the [source manifest](source-manifest.json) (42 exact frozen repository-file hashes), [dependency manifest](dependency-manifest.json) (897 exact pinned NumPy members), and [preserved pre-continuation result](pre-continuation-result.json) (the byte-identical stopped six-session report). These manifests identify the execution sources; they do not claim that all private raw traces or model artifacts are publicly released. Scientific study identity is `c7e10ee9512998b2aaac010753b070b796554d6e72affddd63bdf290d488fea4`; the separately frozen continuation identity is `3d447d273b32ec583c6a6e94b8a6e6f17823e43373a64caa9b08a06e7308c219`.
