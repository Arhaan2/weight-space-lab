# Official final-data audit for the current workflow

The current narrowed study freezes **zero final prompts and zero instructions**.
New untouched final-quality confirmation is unavailable. Development and the
matched standard Browser LLM Bench comparison remain executable; unused final
capacity does not transfer to other stages.

The audit reuses the pinned 541-prompt Google Research IFEval source at revision
`e6890f85757dd84e27ca6df2dd30651dafad28e0`, SHA-256
`67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`.
The existing task-only feasibility rules still admit 344 prompts and exclude
197. No new model generation or expected model performance informs selection.

V1's 128 final prompts and Google item `1000`, which has a saved native smoke
generation, are confirmed exposed. The remaining 412 source prompts have
unresolved historical exposure status. Earlier partition assignments account
for 377 development, 98 validation and 66 final items; an assigned role or an
unexecuted schedule does not establish actual exposure or non-exposure.
The private audit checks available real-model record identities and retains
30 unavailable older record files as unresolved evidence. It does not hydrate,
delete, or reinterpret those files. The earlier exposure audit also records
unverified editor access. No affirmative item-level evidence resolves the
remaining prompts as untouched, so none qualify under the current exclusion
rule. Of the 344 runtime-feasible prompts, 129 are confirmed exposed and 215
remain unresolved.

The implementation supports deterministic subsets of zero through 128 official
items. It preserves source prompts and checker arguments, uses the existing
seed-42 instruction-coverage selector, and excludes unresolved exposure plus
normalized exact and conservative lexical near-duplicates. Sharing an
instruction type alone is not duplication. The private manifest freezes the
ordered IDs, actual counts, rules, source and item hashes, grader configuration,
tokenizer/runtime identities, and evidence audit. Final prompts and later final
results remain outside editor packets. Technical evaluator/editor isolation is
unverified; filesystem location is procedural custody.

The earlier 128-prompt authored V2 set, its fixtures, and its reports remain
historical preparation evidence. It is not silently reused or relabeled as an
official untouched set. Creating a new benchmark is an optional follow-up.

The [sanitized summary](../reports/ifeval-browser-v2-narrowed/final-set-summary.json)
contains the current counts, rules, source identity and manifest digest.
`scripts/freeze_ifeval_v2.py` is the offline audit/seal entrypoint; its
`--verify-only` mode verifies immutable evaluator components. The optional
`--historical-authored` flag preserves reproduction of the superseded authored
preparation. Neither path performs target inference.

Google Research's IFEval authors (Zhou et al.) retain attribution and the
official checker assets retain their Apache-2.0 license.
