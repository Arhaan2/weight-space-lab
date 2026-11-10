# Cleanup and preservation record

The current workflow is [browser quality and resources](ifeval-browser-quality-resource-protocol.md). Cleanup was based on imports, CLI registration, package metadata, subprocess commands, manifests, tests and documented reproduction paths; absence from an import search alone was not grounds for deletion.

## Removed

`scripts/publish.py` was a one-time initial repository creator requiring no existing `origin` and publication from `main`. The repository already exists with a verified origin. Repository-wide references showed only the obsolete publication instructions; no package entrypoint, CLI registration, test, serialized experiment manifest or reproduction command uses it. It was removed and publication documentation now uses a normal topic-branch push and pull request. Its prior source is recoverable from commit `1e9770d0717ce6aeea16d022ef0791349760a53b`.

## Consolidated

The current manual controller owns submission accounting, pristine lineage, conversion, development feedback, selection and continuation. The new resource adapter reuses the pinned original benchmark worker and collectors. Checkpoint validation, recovered virtual-environment launcher, conversion audit, predecessor reconciliation and grader implementations remain canonical helpers rather than copies. Current README and runbook point to one workflow, with old plans explicitly historical.

## Retained for reproduction

The V1 baseline, held-out runner, continuation runner/merger, report/verifier scripts, patches, grades, ledgers and failure records are required by completed experiments and documented replay. Their earlier source remains recoverable from the recorded commits; old source-hash records are not rewritten. A private before/after preservation inventory covers 3,656 historical evidence files.

The earlier V2 random-control routines and authored-set validation are retained for their existing audit/fixture and historical preparation paths. They are outside the current executable schedule, not primary prerequisites. The earlier protocol/configuration and private preparation tree remain unchanged. Existing native/reconstruction implementations and pinned optional dependencies support published replay, recovery, retained CLI commands and tests; no dependency pin was upgraded or removed without evidence of obsolescence.

Pre-existing native-backend/runner edits, historical working documents, untracked scripts/tests/reports, checkpoints, environments, caches and large artifact directories were preserved. Their ownership and replay requirements did not justify deletion. No broad legacy directory or speculative dead-code purge was created.

## New source identity

The narrowed study uses a new root and ledger. Production preparation binds current implementation/configuration and the derived resource preset after offline verification. It records a new study identity without changing old evidence identities. Private instructions, authorization and editor packets remain outside Git; publication includes sanitized summaries only.

Publication inspection found cloud placeholders in the Git object store. Recovery restored 81 objects from local working bytes or the existing public repository only after verifying their exact Git object hashes, and requested macOS materialization of the remaining placeholders. Object identities, refs and history were preserved; no checkout reset or history rewrite was used.
