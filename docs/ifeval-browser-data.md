# Independent IFEval practice data

The `ifeval-browser-v1` study uses Ai2's RLVR-IFeval training data for development
and a separate 128-prompt subset of the pinned Google IFEval source for later
evaluation. It does not reuse or modify the earlier study's Google
development/validation/final partition roles. Familiar instruction types are
intentional; prompt and available task identity, rather than instruction family,
are the separation criteria. The evaluation is an **IFEval subset**, not the
complete benchmark. No evaluation generation is part of this baseline.

`weight_space_lab.ifeval_data` pins Ai2 revision
`47c03c73621c4aab2b824b7818681117d662770e`, the 14,973-row parquet file, and its
SHA-256. Google source revision and file hash come from the existing
`configs/benchmark_sources.json`. Original Google item IDs, prompts, instruction
IDs, and checker arguments are retained. Ai2 identities are immutable file hash
plus row index; the source row's canonical hash and complete ground truth remain
in private provenance.

The adapter explicitly maps 18 Ai2 functions to official checkers. Integer
"at most N" becomes the equivalent official "less than N+1" condition. It
quarantines `around`, arbitrary choice options, ambiguous repeat-prompt metadata,
spaced paragraph separators, exact keyword/letter frequencies, and exact section
counts that do not match the supported official checker schema. It never drops
one constraint from a record or invents a checker. The evaluator remains the
project's pinned Google implementation, with its existing package versions,
sentence tokenizer, and seed-zero random controls.

Selection uses seed 42 in a stable SHA-256 ordering over each source item ID.
The greedy selector first covers rare uncovered instruction types, then balances
type counts, using the seeded ordering to break ties. It selects distinct
normalized available tasks and preserves that order for execution. The frozen
32 development prompts contain 32 instructions across 18 instruction types and
8 top-level families. The 128 evaluation prompts contain 173 instructions across
the same 18 types and 8 families. Evaluation rows containing any unsupported type
are excluded as whole prompts.

Eligibility is determined from task text and constraint metadata before new
model outputs. The exact WebLLM conversation segments are tokenized separately,
including the closed non-thinking prefix. Python counts were checked against the
unmodified tokenizer extracted from the actual installed WebLLM bundle for all
160 selected prompts, without loading an engine or requesting a GPU. Development
inputs use 44–1,644 tokens; evaluation inputs use 42–238. Each leaves the full
1,024-token allowance within the effective 4,096-token context. Conservative
output-size bounds, incompatible output formats, and recorded task-only
contradictions are screened before selection. These estimates do not guarantee
that generation will finish.

Of 14,973 practice rows, 10,410 are eligible and 4,563 are quarantined or excluded.
Of 541 Google prompts, 344 are eligible and 197 are excluded. Detailed reasons,
exact/normalized prompt overlap checks, and available derived-task comparisons
are private. Ai2 does not expose original Tulu task IDs in this source; derived
task comparisons use only exact removal of a recognized inserted directive and
are incomplete for semantic paraphrases. This audit establishes no absence of
model pretraining exposure.

Seven focused adapter tests and 42 official known-answer cases cover all 18
supported functions, including strict pass/fail and integer-boundary examples.
No target forward pass is used by acquisition, mapping validation, selection, or
token counting. The isolated evaluator environment and upstream asset hashes are
recorded in the private experiment directory.

Upstream attribution remains with Allen Institute for AI's Tulu 3/RLVR-IFeval
authors and Google Research's IFEval authors (Zhou et al.). Ai2's dataset card
declares ODC-BY and includes additional source terms; all supplied license files
are preserved privately. Google's checker source and Apache-2.0 license are
preserved unmodified. This extension does not replace the project's existing
scientific authorship or Gufran's contribution.
