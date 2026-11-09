# Final matched held-out IFEval subset evaluation

Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

The separate `ifeval_heldout_runner` schedules the previously frozen 128 Google IFEval prompts and 173 checked instructions against the selected development candidate and pristine model. It preserves the existing development sources, raw adapter, official grading, extraction, runtime, and evidence. The small [browser scheduling patch](../integrations/ifeval-heldout-schedule.patch) adds the explicit held-out stage without altering worker generation behavior.

Before any inference, the freeze verifies the four existing source/deployment identities, unchanged tokenizer/template/model library, original data identities and order, proposal provenance, and fresh authorization. Both source identities use the canonical complete saved-file map, as in the original study. No proposal import or model conversion is performed.

Execution is pristine first, then candidate. Each arm owns one load, one warm-up, 128 scored requests, 129 dispatches and at most 131,088 generated tokens. Independent durable per-arm ledgers enforce immutable halves of the aggregate 258-dispatch and 262,176-token authorization; unused capacity cannot transfer. A shared clock bounds first load through final cleanup to 180 minutes, with 90 minutes per arm and 120 seconds per generation. The candidate cannot start until pristine execution and unload/process cleanup are verified. Execution destinations and the single-run intent are exclusive, so the commands cannot automatically resume or retry.

`ifeval_heldout_analysis` retains every scheduled item, using the existing paired prompt bootstrap and an instruction-count ratio bootstrap that resamples whole prompts. Unexecuted slots, ambiguous dispatches, technical failures and grader errors remain distinct. Official grades are never replaced by qualitative interpretation. The frozen analysis uses 10,000 resamples with seed zero; no exact paired significance test exists in the existing implementation.

Private authorization, prompts, outputs and machine paths remain in the ignored artifact destination. The [final report](../reports/ifeval-browser-v1-heldout/README.md) contains only sanitized results, identities and interpretation. The evaluation is a 128-example IFEval on-device subset, not a full 541-example benchmark run. Its final result closes this candidate evaluation, irrespective of direction or uncertainty. The subset is exposed afterward and cannot be called untouched for future edits.
