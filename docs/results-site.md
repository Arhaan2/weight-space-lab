# Public results site

The buildless site in `site/` presents the measured V1 comparison and the actual
status of the V2 follow-up. It runs on GitHub Pages without a framework, package
installation, external analytics or a model download. Only the contents of
`site/` belong in the Pages deployment.

## Evidence

`site/results.json` is a sanitized view of the public experiment reports. The
page retains readable development, historical held-out and recorded resource
tables without JavaScript. JavaScript adds artifact, scope and metric controls. When updating results, keep that fallback and the
JSON consistent. Link historical evidence to its immutable public commit,
because an open research pull request may be ahead of `main`.

Before publication, run from the repository root:

```sh
python3 scripts/update_results_site.py
python3 scripts/update_results_site.py --check
python3 scripts/verify_results_site.py
node --check site/app.js
python3 -m http.server 8080 --bind 127.0.0.1 --directory site
```

The evidence verifier compares all four held-out metrics, paired prompt outcomes,
development results, token/duration accounting and allocation observations with
the canonical public report. It also checks V2 against its public execution
report and independently checks each evaluated candidate's qualification rule.
The update script synchronizes V2 and the independent resource study from their
public reports. It embeds script-safe copies of results and methodology, updates
the no-JavaScript tables, and normalizes public source URLs.
The verifier checks the deployment file types and scans
text files for local paths and common credential markers. It does not execute a
model or independently regrade private outputs.

The resource checks independently recompute request totals, session means,
ranges and baseline deltas, enforce missing-data and matched-environment rules,
and compare all published study-source files to the frozen source manifest.
Run the final verification in the clean release checkout. Unrelated local edits
to a frozen source file correctly fail that check and must not be overwritten to
make a development workspace appear clean.

Review the rendered page at desktop, tablet and phone widths, with keyboard
navigation and reduced motion. Stop preview/inspection browsers before running
the target-model resource measurements on the same machine.

## Publishing

Use a topic-branch pull request for source changes. Keep existing research
branches and unrelated working changes intact. A separate `gh-pages` deployment
branch contains only the reviewed files from `site/`, mapped to its root; set
GitHub Pages to that branch and `/`. This lets the public presentation deploy
without merging unrelated experimental implementation changes.

The deployment includes `.nojekyll` and uses relative asset URLs, so it works
under the repository path. Preserve the font's included OFL license. Verify the
Pages build reaches `built` and the served HTML/data match the reviewed content
before reporting a live URL. Never copy private packets, responses, checkpoints,
authorization, ledgers or host process logs into `site/`.

GitHub's [Pages REST documentation](https://docs.github.com/en/rest/pages/pages)
defines the source-branch setting and build-status checks.

## Design and review

The current design is a research workbench, rebuilt after the user rejected the
previous colorful landing page. Its **Achromatic ledger** app direction uses seed
`1571155384`; the explicit professional brief supersedes the prior orange/lime
palette and oversized typography. The prior screenshots and critiques remain a
before record, not approval of this replacement.

Taste's full app-screen workflow supplies flat neutral surfaces, dense evidence,
right-aligned tabular figures and a single muted teal on the selected data. The
face is a restrained serif research title with the existing self-hosted Schibsted
interface font. The product-specific element is a five-artifact comparison and an
exact weight-edit map: actual affected column intervals are positioned within the
verified 3,584-column down-projection dimension. No decorative weights are drawn.

Three hash routes answer separate questions: **Results** compares artifacts;
**Methodology** explains how measurements were obtained; **Evidence** supplies
reports, identities and downloads. Selection composes the chart, selected reading,
paired baseline difference, edit operations and individual resource sessions. Each resource session also exposes its actual
non-warm-up request records through a keyboard-accessible range selector and
three 20-request trace windows. Timing and token views share exact observations;
horizontal spacing is request order, with observed idle gaps listed separately.
Selected-session CSV exports only public numeric/finish metadata. V1
held-out quality, the reused 32-prompt development set and the independent resource
pilot never share a statistical denominator. Resource means require two completed
60/60 sessions with available measurements; partial records remain inspectable.

Apple guidance supplies native keyboard controls, persistent focus after selected
rows update, visible focus indicators, immediate local feedback, destination
heading focus on route changes, and reduced-motion support. No entrance motion
or external model call is used. Printing methodology temporarily expands details
and restores their prior state. The complete numerical fallback and source links
remain usable without JavaScript.

Final review used a clean publication checkout after all ten target sessions
ended. Taste app inspection passed with **zero failures** at 1,440, 768 and
390 pixels. All six images in each of two rounds were visually examined.
The second round kept the phone metric selector at full width, set its measurement
unit as one unbroken line below the number, and shortened the adjacent continuation
notice while retaining the complete recovery record in Methodology.

Two inspector warnings are intentionally retained. The nearly neutral `#f7f7f4`
ground implements the requested restrained research palette. At phone width,
tables become per-artifact cards with labels over values; those fields are
left-aligned within each card, while desktop numeric columns and chart readings
remain right-aligned. Neither width has horizontal overflow. Text contrast checks,
asset loading and browser exception checks pass. Neutral bars meet 3:1 graphical
contrast and the selected teal remains distinct.

The functional browser harness passed artifact and metric comparisons against the
saved evidence, exact edit maps, session/request selection, warm-up exclusion,
request and aggregate CSVs, keyboard focus, shared views, methods printing,
citation download, reduced motion, and no-JavaScript fallback. Static JavaScript
syntax, Python compilation and updater idempotence also pass. No review browser
ran during target measurement, and no frozen report or benchmark source changed
as part of interface review.

Final visual critique: design 8.4, usability 8.3, creativity 8.0, content 9.1;
overall approximately 8.4/10. The main comparison, actual weight-column locations
and request traces give the interface a specific research purpose. Dense caveats
remain in expandable methods; the resource view preserves its interruption notice
and the visibly slower second R3 observation. These scores assess presentation,
not scientific strength.

Review artifacts are `.taste-skill/research-upgrade-2/desktop-fold.jpg`,
`desktop-page-01.jpg`, `tablet-fold.jpg`, `tablet-page-01.jpg`, `mobile-fold.jpg`
and `mobile-page-01.jpg` in the publication checkout. The structured inspection
record is `report.json`; the parent task retains the functional harness output.

The updater accepts `--repository-base https://github.com/Arhaan2/weight-space-lab/blob/COMMIT/`
to pin newly published methodology/source links, `--source-url` for the V2 report,
and `--resource-source-url` for the independent resource report. Preserve historical
immutable pins. `--check` performs no writes and confirms JSON, embedded data and
static tables remain synchronized.
