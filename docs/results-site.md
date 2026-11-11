# Public results site

## Current mechanistic harness view

The default `#harness` route presents the updated workflow separately from the
completed studies. It keeps IFEval as the only accuracy benchmark and describes
one pristine and one frozen-candidate browser evaluation, with compliance and
resource observations joined on the same requests. Diagnostic probes have their
own accounting. The weight/activation/direction/evaluation explorer is keyboard
accessible, retains focus when stages change, and has complete script-free text.
CPU Transformers source-model activations are not described as WebGPU activations.

The public contract lives in `site/harness.json`. `HARNESS_CONTENT` and
`HARNESS_DATA` are updater-controlled regions; no historical report is changed.
The initial state is `implemented_not_run`, with an explicit empty paired table.
The optional `--joint-report REPORT.json` imports the canonical
`mechanistic_study report` output into a separate `joint-comparison.json`. It
whitelists aggregate numerics, hashes, bounded item identities, grades and request
observations; free-form provenance, model answers and prompts are excluded. The
updater checks report state, per-arm denominators, pass counts, finite numerics
and request order for completed pairs, but does not independently regrade or
authenticate a source capture. A later update reuses the saved public report.
Arm status, grades, response cost, native peak, cost per strict pass and expandable
coverage/request tables are generated as ordinary HTML, including without scripts.
Paired changes require the canonical report's completed, matched comparison;
partial records cannot borrow historical values. Missing results are unavailable,
never zero. The workflow contract's evaluation status and result link follow the
imported record while historical evidence stays separate.
`source_base` identifies the implementation release separately from historical
evidence pins. Preserve or update that link when publishing the matching source.

The existing Achromatic ledger style, type and palette are retained. Four
navigation links fit at phone width, stage buttons remain at least 44px tall,
and the same-request measurement contract is readable without JavaScript. The
historical Results, Methodology and Evidence routes keep their original data
and separate-study denominators.

The harness extension was checked at 1,440, 768 and 390 pixels with reduced
motion, keyboard stage selection, destination focus, historical navigation and
script-free fallback. Taste's app inspection found zero failures. Three retained
warnings reflect the established neutral palette, a protocol explorer occupying
the initial view, and phone table values beneath their labels. Both inspection
rounds' six images were reviewed; the second used the preserved Achromatic ledger
app direction. A separate disposable fixture site exercised the future imported
record at all three widths, including coverage and request disclosure, without
publishing fixture numbers or launching target inference. No horizontal overflow
or browser errors were observed. Review records are under
`.taste-skill/harness-design/`; these are presentation checks, not benchmark data.

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

The three historical hash routes answer separate questions: **Results** compares artifacts;
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

The prior five-artifact revision was reviewed in a clean publication checkout
after its ten historical target sessions ended. Taste app inspection passed with **zero failures** at 1,440, 768 and
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

Prior revision’s visual critique: design 8.4, usability 8.3, creativity 8.0, content 9.1;
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

## Fresh-baseline comparison extension

The six-artifact comparison preserves the Achromatic ledger direction (seed
1571155384). Its main question is whether a measured quality change coincides
with a resource tradeoff. The new **All versions** view places all available
artifacts on one quality-versus-resource figure, with selectable strict/loose
quality and twelve resource metrics, a selected-version reading and a complete
score/resource table. Points and table rows share the selected artifact. Exact
coincident values share a numbered native button that cycles through their
versions. Small dots remain at the exact measured coordinates; connected labels
are placed independently to keep their 44px targets clear of nearby labels.
No jitter or synthetic separation changes the observations. Focused
axes are explicitly labeled and can switch to full scales beginning at zero.

A cohort selector separates fresh baseline measurements from historical values.
Fresh observations are never filled from old runs. Missing quality or either
resource replicate yields an unavailable paired figure; resource session values
remain inspectable in their detail view. Fresh baseline deltas require both
quality and resource environment checks. CPU/GPU counters remain activity rates,
not model-only FLOPs or energy. Activity-rate views display measured response and profile durations alongside the selected observation; both durations are also selectable plot metrics. These durations are context, not a model-only integration interval for the counters. The historical figure retains its Chrome-version
and separate-study caveats.

The data contract is `results.json.fresh_comparison`, an unchanged copy of the
new public report. Artifact metadata use `key` or `artifact_key`; the fourth
Astra artifact is `astra-r4`. Quality rows live under `quality.artifacts`, and
resource sessions retain the existing session schema under `resource.sessions`.
The report does not add placeholder fourth-round results to the older cohorts.
`FRESH_EVIDENCE` and `FRESH_METHODS` regions are filled by the release updater for
script-free evidence and complete on-page fresh methodology.

Six bibliographic references now sit at the bottom of every view: the motivating
Sparks of Weight-Space Programming report, the IFEval paper and pinned checker,
Ai2 RLVR-IFeval, the pinned Qwen model, the WebLLM paper/runtime and Browser LLM
Bench. Titles and authors for the motivating report, IFEval and WebLLM were
checked against their primary pages. Study narratives and methods are readable
on the website; report actions lead to on-page sections or exact JSON, not a
README. The existing repository link remains available for implementation code.

The final six-artifact interface was reviewed only after all six quality sessions
and twelve resource sessions ended and their cleanup passed. Both browser QA
rounds used the final canonical measurements in the clean publication checkout;
the final round also used the immutable scientific source pins. The prior
five-artifact review above remains a separate historical record.

The final Taste app inspection reports **zero failures** at 1,440, 768 and
390 pixels. All six images from each of two rounds were opened and visually
examined. The second round shortened native-select options, removed the duplicate
phone cohort label, and used 24px phone readings with the percentage unit inline.
The near-point fix keeps small dots at measured coordinates and separates only
the connected 44px label controls. Full axes and focused axes retain the same
values; exact coincidences remain keyboard-selectable groups.

The full interaction harness passed historical and fresh quality/resource values,
all six artifacts, all twelve resource metrics, both scales at all three widths,
nonoverlapping plot targets, coincident-point keyboard cycling, paired fresh
outcomes (three improved, three regressed, 26 tied for R4), session/request traces,
CSV exports, shared links, printing, six references, and no rendered README links.
No-JavaScript review verified the fresh six-row fallback, R4's 16,998 quality
tokens, frozen prediction and cache-maintenance methodology. Direct-file rendering
and the fresh methodology were also checked at desktop and phone widths. No
browser exceptions, missing local requests or horizontal overflow were observed.

The two retained inspector warnings have the same rationale as before: the
nearly neutral ground belongs to the established direction; phone numeric fields
sit beneath their own labels in per-artifact rows, while desktop columns remain
right-aligned. There is no synthetic data or stock imagery. The interface remains
dense on a phone because the selected metrics, limitations and six complete
artifact records are retained.

Final app critique: design 8.5, usability 8.5, creativity 8.0 and content 9.2;
weighted presentation score approximately 8.5/10. The all-version figure now
answers the quality–resource question without losing coincident versions, and
the complete methods and references stay on the website. These scores concern
presentation, not scientific validity. The final screenshots and inspection
record are under `.taste-skill/fresh-r4-review-2/` in the publication checkout;
functional results and full-scale screenshots are under
`.taste-skill/research-interactions/`.

## First measured joint result — November 11, 2026

The live Harness view now leads with the actual 32-prompt comparison, followed by
the edit and workflow. The canonical result is
`reports/mechanistic-browser-v1/joint-comparison.json`; `intervention.json` binds
the selected columns, source/effective quantization audits, diagnostic accounting
and unchanged pre-evaluation rationale. Import with both `--joint-report` and
`--intervention-report`, then run the updater check and evidence verifier.

Strict compliance fell from 16/32 to 15/32, output tokens increased 1.25%, and
observed total response time decreased 2.97%. Both tokens and response time per
compliant answer increased. The page explicitly says the joint objective was not
met; it does not turn this one timing observation into a causal speedup claim.
It retains partial telemetry coverage and unavailable energy.

The first candidate Chrome launch ended with zero model calls. A preserved,
explicit pre-load continuation completed the same frozen candidate's first
evaluation. The public record retains the zero-call proof, 6.120 seconds of
pre-load supervisor time, and distinct browser-launch/model-load counts.
Historical comparison files remain byte-for-byte unchanged.
