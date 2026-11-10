# Manual browser quality and resource workflow

Use this runbook with the [current protocol](ifeval-browser-quality-resource-protocol.md), [configuration](../configs/ifeval_browser_quality_resource.json) and [preparation report](../reports/ifeval-browser-v2-narrowed/README.md). The earlier control-and-authored-set preparation remains in its original private root. Do not reopen a V1 ledger or use the old V2 packet.

## Canonical command

Run from the repository root with the real pinned evaluator environment:

```bash
study_python=artifacts/ifeval-browser-v1/evaluator-env/bin/python
study_root=artifacts/ifeval-browser-v2-narrowed
study() { "$study_python" -m weight_space_lab.ifeval_v2_study "$@" --root "$study_root"; }
study status
```

`status` performs read-only identity checks. The module is the canonical command path; the small `scripts/run_ifeval_v2.py` wrapper delegates to the same parser. Do not dereference a virtual-environment interpreter to system Python. The converter uses its separately pinned `artifacts/browser-integration-v1/conversion/venv/bin/python` interpreter.

## Preparation and freeze

Complete code changes and offline verification first. Audit official final-data exposure, verify pristine and saved development evidence, derive the resource preset from the pinned standard profile, and freeze current source/configuration hashes. `prepare` creates a fresh study and binds the already supplied private study authorization; it does not run inference:

```bash
study prepare --authorization private/study-authorization.txt
study export --round 1
```

For this delivery, those commands have already been run against the recorded private authorization; use the delivered packet and existing state. Never rerun preparation over an existing study. `authorize` remains a low-level binding command for an independently reconstructed study, not a new approval step at each handoff.

The first production packet export anchors the seven-day calendar once. Re-export/restart does not reset it. Implementation time and manual waiting do not consume active local processing, but manual waiting consumes the calendar window.

## Genuine Astra responses

Open only `artifacts/ifeval-browser-v2-narrowed/editor/round-1/editor-packet.txt` in the authenticated manual Astra / Ultra interface. It contains real request identifiers and hashes, all development inputs/outputs/grades, relevant observations, approved weight information and the exact response schema. Do not send the evaluator directory, final material, private authorization or other session transcripts.

Before submitting the packet, record the proposal slot:

```bash
study begin-turn --round 1
```

Save Astra's original returned JSON without substituting a fixture or prefilled proposal, then continue the authorized local stages:

```bash
study advance --response "$study_root/editor/round-1/response.json"
```

The controller imports and audits the response, applies its edit to pristine, converts and packages it, evaluates the actual deployed candidate on 32 development tasks, and exports the next packet. Repeat only for the actual next round requested, up to four submissions. After the last eligible round, the same continuation freezes selection and executes available final-task and A–B–B–A resource comparisons. No additional general execution approval is required.

Invalid, expired or oversized responses consume the slot. Abandonment is explicit and cannot reclaim it:

```bash
study close-turn --round 1 --reason operator_abandoned_without_replacement
study advance
```

Do not run that example on a live round unless it was actually abandoned. Duplicate proposals/artifacts cannot create replacement capacity. Do not perform extra smoke generations, calibration, retries, model loads or warm-ups.

## Results and recovery

`advance` stops for the next genuine manual input or an actual integrity/runtime/budget failure. `status` reports the durable state; completed work and consumed capacity survive restarts. The low-level `import`, `build`, `evaluate`, `select`, `final` and `resource` commands exist for inspection and bounded operation, not extra attempts. Never edit state or delete reservations to resume.

```bash
study status
study recover
```

Use `recover` only after an interrupted controller: it reconciles an unresolved stage conservatively and never redispatches uncertain work. Unconfirmed cleanup stops the study. Preserve logs, partial records and failed attempts. Frozen source changes invalidate execution; do not loosen that guard.

After execution, `study analyze` produces the final task report with actual denominators, paired metrics and limitations. Resource reporting preserves individual sessions and their original benchmark reports. With zero eligible untouched final prompts, there is no new final-task comparison. With pristine selected, both comparisons are skipped; no pristine-versus-pristine experiment is run. Publish V2 findings only after their corresponding runs complete.

## Budget and interpretation

The authorization ceiling is four submissions, four candidate imports/conversions/packages, ten loads/warm-ups, 384 task requests plus 240 resource requests, 634 dispatches and 408,736 target tokens. Development/final/resource browser clocks are 60/90/30 minutes inside three browser hours and four active local hours, with 32 GiB retained storage. The actual final-set size reduces this schedule; unused final capacity cannot be reassigned.

Selection requires complete valid development, strict prompt gain over 16/32, at most 13,941 scored output tokens, and frozen deployment/memory compliance; rank by strict score, then tokens, then earlier round. A quality tie retains pristine. Keep nonqualifying trade-offs and timing/memory uncertainty visible.

IFEval task output is not the original Browser LLM Bench resource workload. The latter preserves the pinned 60-request standard session, original windows/idle gaps, 64-token ceiling and native telemetry semantics. Two sessions per artifact support a small measurement pilot, not precise equality. Technical editor isolation and unexposed provider metadata remain unverified/unavailable.
