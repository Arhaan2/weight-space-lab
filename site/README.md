# Weight Space Lab research workbench

Buildless GitHub Pages site. Serve this directory with a static HTTP server; no
package installation, model download or build is required. Only this directory
belongs in the Pages deployment.

Three views separate interactive results, methodology and evidence. Select a
weight artifact, quality scope or resource metric to inspect actual observations,
paired baseline differences, edit operations and individual sessions. Historical
V1 held-out quality, development quality and the independent resource pilot are
never pooled. Missing measurements remain unavailable; resource means require two
completed sessions with available readings. There are no synthetic results or AI
backend calls.

`results.json` preserves the public V1/V2 evidence and includes the full sanitized
independent resource report when available. `research-methodology.json` describes
scopes, generation, scoring, provenance and telemetry semantics. The updater embeds
both JSON files safely in `index.html`, alongside generated no-JavaScript tables.
The evidence view links complete data, source reports and `citation.cff`.

Run `python3 scripts/update_results_site.py` from the repository root, then use
`--check` for read-only synchronization verification. Source-link flags are
`--source-url`, `--resource-source-url` and `--repository-base`. No inference or
browser is launched by the updater. Perform visual checks only when target resource
measurements are idle.

Design: Taste's app-screen Achromatic ledger, seed 1571155384, adapted to the user's
explicit restrained professional direction. Apple guidance supplies feedback,
focus, navigation, reduced motion and readable fallback behavior. Existing Schibsted
Grotesk is self-hosted under the included SIL Open Font License in `assets/OFL.txt`.
No private prompts, responses, editor packets, machine paths, credentials or
session transcripts are included.
