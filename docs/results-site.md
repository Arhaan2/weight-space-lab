# Public results site

The buildless site in `site/` presents the measured V1 comparison and the actual
status of the V2 follow-up. It runs on GitHub Pages without a framework, package
installation, external analytics or a model download. Only the contents of
`site/` belong in the Pages deployment.

## Evidence

`site/results.json` is a sanitized view of the public experiment reports. The
page retains a readable V1 comparison without JavaScript; JavaScript adds the
split and scoring controls. When updating results, keep that fallback and the
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
The update script synchronizes V2's JSON and static HTML from that public report.
The verifier checks the deployment file types and scans
text files for local paths and common credential markers. It does not execute a
model or independently regrade private outputs.

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

The taste direction is **Type and nothing else**, seed `51409817`, using
Schibsted Grotesk and the dealt orange, sky and lime fields. No direction was
redrawn. The measured difference between 69 and 70 held-out passes supplies the
main comparison; its uncertainty remains beside it. The evidence figures use
the **Accent ladder** interface treatment, seed `1092602209`, with the page's
type and colors. Five core type roles and one data-display role hold the scale.

Apple interaction guidance supplies immediate press feedback, visible keyboard
focus, named controls, reduced-motion behavior, and readable static fallbacks.
The one live interaction switches actual V1 split/scoring combinations; it
never simulates an experiment. V2's generated table remains present without
JavaScript and distinguishes development evidence from fresh confirmation.

The final third review inspected the completed V2 table at 1,440, 768 and
390 pixels. All ten screenshots from the final verification were reviewed.
The inspector reported zero failures. Its three warnings are documented:
photographs and footage would not add experimental evidence; the longest
context paragraph remains a readable, bounded 78ch block alongside the wider
results table (the inspector estimates about 95 average characters per line).
It wraps naturally on narrow screens. No decorative media was added to fill
those warnings.

The final pass corrected a page-sized interface annotation by placing it on
the actual table, and revealed the V2 heading, table and notes individually.
Keyboard activation, all four comparison states, pressed-state labels, reduced
motion, phone overflow, anchors, no-JavaScript reading and browser exceptions
passed. No inspection browser ran during target measurement. The evidence
verifier passed 39 checks against the canonical reports.

Final critique: design 7.9, usability 8.2, creativity 7.5, content 9.0; weighted
overall 8.02/10. Remaining density in the research caveats is deliberate: the
browser-version difference, exposure limits, selection failure and attribution
are necessary context for interpreting the measurements.
