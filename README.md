# Weight Space Lab research workbench

Buildless GitHub Pages site. Serve this directory with a static HTTP server; no
package installation, model download or build is required. Only this directory
belongs in the Pages deployment.

The default **Harness** view describes the current weight/activation inspection,
directional-edit and same-request browser evaluation workflow. IFEval is the
single accuracy benchmark. One pristine and one frozen candidate run supply both
the grades and resource measurements; bounded diagnostic probes are separately
accounted. Source Transformers activations are explicitly distinguished from
quantized browser execution. The new paired result remains unmeasured.

Three additional views separate completed results, methodology and evidence. Select a
weight artifact, quality scope or resource metric to inspect actual observations,
paired baseline differences, edit operations and individual sessions. Historical
V1 held-out quality, development quality and the independent resource pilot are
never pooled. Missing measurements remain unavailable; resource means require two
completed sessions with available readings. There are no synthetic results or AI
backend calls.

The fresh cohort contains pristine, V1 and all four Astra edits. Its All versions
figure compares strict or loose compliance against twelve selectable resource
metrics. Small dots retain exact measured coordinates; connected control labels
remain selectable when results are close. Fresh and historical cohorts stay
separate, and both resource sessions remain inspectable. Methods and six primary
references are readable on the website without a README detour.

`results.json` preserves the public V1/V2 evidence and includes the full sanitized
independent resource report when available. `research-methodology.json` describes
scopes, generation, scoring, provenance and telemetry semantics. The updater embeds
both JSON files safely in `index.html`, alongside generated no-JavaScript tables.
The evidence view links complete data, source reports and `citation.cff`.
`fresh-comparison.json` and `fresh-maintenance.json` retain the new canonical
comparison and its separately bound cache-maintenance record unchanged.

Run `python3 scripts/update_results_site.py` from the repository root, then use
`--check` for read-only synchronization verification. Source-link flags are
`--source-url`, `--resource-source-url`, `--fresh-source-url`,
`--fresh-maintenance-source-url` and `--repository-base`. No inference or
browser is launched by the updater. Perform visual checks only when target resource
measurements are idle.

`harness.json` is the versioned public workflow contract, separate from historical
evidence. The updater renders its stage explorer, no-JavaScript explanation,
measurement definitions and awaiting-results table into `index.html`, and embeds
a script-safe copy. Import the first canonical paired report with:

```sh
python -m weight_space_lab.mechanistic_study report --root STUDY --output REPORT.json
python3 scripts/update_results_site.py --joint-report REPORT.json
python3 scripts/update_results_site.py --check
```

The updater writes `joint-comparison.json` using an explicit public field list,
then renders arm totals, cost per strict pass, telemetry coverage and each request.
Prompts, answers, machine paths and arbitrary execution fields are omitted. Partial
arms stay visible; paired deltas require a completed matched comparison in the
canonical report. Missing measurements stay unavailable. Later updates reuse this
public report without needing the import flag. This is publication validation,
not an independent regrade or authentication of the original capture. Source links
use the contract's `source_base`; pin it to the actual published implementation
when releasing the harness.

Design: Taste's app-screen Achromatic ledger, seed 1571155384, adapted to the user's
explicit restrained professional direction. Apple guidance supplies feedback,
focus, navigation, reduced motion and readable fallback behavior. Existing Schibsted
Grotesk is self-hosted under the included SIL Open Font License in `assets/OFL.txt`.
No private prompts, responses, editor packets, machine paths, credentials or
session transcripts are included.
