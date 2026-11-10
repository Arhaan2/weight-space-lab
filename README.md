# Weight Space Lab results site

Buildless GitHub Pages site. Serve this directory with any static HTTP server; no package installation or build is required. Only this directory belongs in the Pages artifact.

`results.json` contains a sanitized projection of the saved public V1 report, with immutable source links. The default comparison and all explanatory text are present in `index.html`, so the evidence remains readable without JavaScript. `app.js` adds the comparison controls and reads the current V2 description from the data file. Keep static V2 copy and the data file aligned when publishing a new measured result.

The live comparison selects the actual V1 development/held-out prompt set and strict/loose scoring. The paired-outcome and confidence-interval figures below it are explicitly the strict held-out result. V1 resource figures are observational held-out-task measurements. V2 standard Browser LLM Bench session tables are generated only when the public execution report contains measured sessions; otherwise the page states whether the comparison is pending, stopped or skipped.

No private editor packets, responses, study authorizations, machine paths or credentials are included. Source and attribution links point to the public evidence and upstream projects. Font files are self-hosted Schibsted Grotesk from Google Fonts and include their SIL Open Font License in `assets/OFL.txt`.

Design: taste’s “Type and nothing else,” seed 51409817, with its Schibsted and orange/sky/lime palette; evidence fragments follow “Accent ladder,” seed 1092602209. Apple interaction guidance supplies immediate press feedback, keyboard focus, accessible labels, reduced-motion behavior and readable fallbacks. The scientific evidence figures are the imagery. No stock photographs, generated media or sample evidence is used.

To refresh measured V2 evidence from the sanitized public report, run `python3 scripts/update_results_site.py` from the repository root. Add `--check` for read-only verification or `--source-url` to pin the public execution-report link. The updater changes only `site/results.json` and three explicitly marked HTML regions, keeping the no-JavaScript table, live description and conditional resource status aligned. It runs no inference or browser.
