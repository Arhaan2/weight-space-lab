# Publication and provenance

## Repository identity

New project commits use the maintainer's verified account-linked email and a neutral branch name:

```bash
git config --local user.name "Arhaan Aggarwal"
git config --local user.email "59648536+Arhaan2@users.noreply.github.com"
```

Before publishing, inspect both author and committer metadata:

```bash
git log --format='%h %an <%ae> | %cn <%ce> | %s'
git status --short
git diff --cached --stat
```

Use `main` and descriptive topic branches. Do not add automated coauthor trailers. When another person contributes, retain their legitimate authorship and permission; this project's attribution policy does not authorize rewriting someone else's history.

This repository already exists. Publish reviewed changes through a descriptive topic branch and a pull request to `main`; preserve local history and push normally:

```bash
git switch -c ifeval-browser-quality-resource
# Stage only the reviewed files belonging to this change.
git diff --cached --stat
git diff --cached --check
git commit -m "Implement bounded browser quality and resource study"
git push -u origin HEAD
gh pr create --base main --title "Implement bounded browser quality and resource study" --body-file /tmp/weight-space-release.md
```

Verify the destination is `Arhaan2/weight-space-lab` and the authenticated account is `Arhaan2`. Inspect all outgoing commits and blobs, including historical versions, for private content before pushing. Do not force-push or rewrite history. Preserve unrelated working files outside the staged change. The obsolete one-time repository-creation script was removed because its required state (no origin, initial publication from main) no longer applies; its exact source remains in earlier commits.

## Public content

Publish original implementation, test fixtures, experiment specifications, task formats, patch artifacts, scientific provenance, reproduction commands, and measured results. Identify Astra as the experimental editor when it was actually used, including the model/version exposed by the execution interface. Git authorship and experimental model provenance describe different things.

Keep development-session instructions, assistant transcripts, private environment files, access tokens, raw tool dumps, and local machine paths outside the repository. Prefer relative paths in code, documentation, and reports. Review the tracked file list and staged contents rather than assuming an ignore file removes material already tracked.

Do not publish an empty result table with invented scores. Label synthetic examples as synthetic. Record pending experiments as pending. Preserve upstream citations, license notices, and the original benchmark's authorship.

## Before a result release

- Verify the exact base and edited checkpoint hashes and their saved/reloaded scalar differences.
- Freeze evaluation code, dataset identities, chat template, sampling, dtype, runtime, and hardware metadata.
- Preserve every scheduled item and account for missing, failed, truncated, and invalid generations.
- Show all predeclared controls, accepted and rejected candidate counts, budget use, and regressions.
- Keep experimental test isolation explicit; release withheld task material only after selection is frozen and permitted disclosure has been checked.
- Verify a clean checkout and independently rerun the published analysis.
