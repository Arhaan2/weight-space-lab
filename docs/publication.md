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

If starting from a source archive, initialize the repository before setting its local identity:

```bash
git init -b main
git config --local user.name "Arhaan Aggarwal"
git config --local user.email "59648536+Arhaan2@users.noreply.github.com"
git add README.md LICENSE pyproject.toml .gitignore configs docs examples scripts tests weight_space_lab
git diff --cached --stat
git commit -m "Initialize controlled weight-editing study"
```

With the GitHub CLI installed and signed in as Arhaan2, `python scripts/publish.py` creates the public `Arhaan2/weight-space-lab` repository and pushes the reviewed commit. It checks account, commit identities, coauthor trailers, branch, working tree, and existing remote before publication. It never makes up benchmark results or claims the destination already exists. Do not run it if this repository has already been created; inspect the existing destination and set its remote explicitly instead.

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
