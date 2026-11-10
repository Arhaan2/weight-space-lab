# Browser integration v2 runtime adapter

The v2 checkout is `artifacts/browser-integration-v2/benchmark`. It is copied from the corrected browser pilot checkout. The v1 integration and pilot checkouts remain historical evidence. This adapter does not authorize attempts; the separate supervisor owns the new ledger, authorization, shared deadline, and sequential arm orchestration.

The source-only patch is `integrations/browser-integration-v2.patch`. It includes the reused task adapter and correction, new fetch-only preflight, durable dispatch gates, driver, and offline tests. It excludes model bytes, generated bundles, private prompts, manifests, captures, and results. Arhaan2's existing authorship is preserved.

## Fetch-only preparation

The frozen configuration includes `servedBuild`, an array of `{url, sha256, kind}`. `kind` is `html` or `javascript`. Include the served HTML and client and every JavaScript file in `public/benchmark-worker/`. The driver checks their local hashes before launching a dedicated browser profile. The browser then fetches and validates the actual origin's bytes, HTTP status, MIME, JSON parsing, WASM magic, SHA256 identities, JavaScript import closure, configuration/tokenizer/shard/WASM resolution, and artifact-specific model/cache URLs. HTML fallback responses are rejected.

`node scripts/run-task-evaluation.mjs CONFIG OUTPUT --fetch-only --code-version-sha SHA` performs this preparation without creating a Worker, WebGPU adapter/device, or inference engine. A ledger and attempt ID are not needed for this mode. Its `fetch-preflight.json` reports `status: passed`, `fetch_only: true`, and zero `engine_loads`, `generation_dispatches`, `workers_created`, and `gpu_requests`, with per-file evidence. Those zeros describe this preparation only.

## Durably gated attempts

`node scripts/run-task-evaluation.mjs CONFIG OUTPUT --attempt-id ID --code-version-sha SHA` starts one attempt. It repeats fetch-only browser validation, then creates the task worker. It never retries internally. The driver supplies gate metadata separately from the frozen generation configuration, avoiding a configuration/source-hash cycle. Optional `--load-only` disallows generation; it is not a successful integration result.

Before `CreateMLCEngine`, the worker emits a load gate request. Before the sole `chat.completions.create`, it emits a generation gate request. The driver atomically publishes and fsyncs `load-intent.json` or `generation-intent.json`. Each has schema `browser-v2-dispatch-intent` and fields `kind`, `arm`, `request_id`, `attempt_id`, `code_version_sha256`, `configuration_sha256`, `artifact_sha256`, `run_id`, and `created_at_epoch_ms`. Request IDs are `load-1` and `generation-1` within an attempt directory.

The supervisor first commits the corresponding ledger entry, then atomically publishes and fsyncs `load-ack.json` or `generation-ack.json`. An approval must include matching `kind`, `arm`, `request_id`, `attempt_id`, and `code_version_sha256`, `approved: true`, and the unchanged absolute `deadline_epoch_ms`. No valid approval means no inference API call. The worker records receipt and a call-intent event immediately before the call. An acknowledged or ambiguously forwarded generation remains consumed regardless of output.

The absolute deadline re-arms worker cancellation and termination with cleanup reserves. The supervisor's independent deadline covers browser/CDP failures. Each attempt uses a fresh isolated browser profile, runs sequentially under the supervisor, unloads its engine when available, terminates the worker, and closes its dedicated Chrome and harness processes. Process exit evidence is distinct from engine unload confirmation. No unrelated browser caches or profiles are cleared.

## Evidence and failure handling

Loading captures automatic initialization GPUQueue submissions and observed WebGPU devices separately from application generation. Zero application requests does not imply zero computation during loading. Actual generation additionally requires positive observed queue submissions. Task outputs preserve raw text, natural finish/truncation, tokens, timing, resource scope, and errors; grading remains the existing offline task grader.

Only a concrete asset guard error before generation and without a resource stop receives `confirmed-pre-generation-asset-routing-or-fetch`. Its event retains URL, kind, and original error. This is evidence for supervisor diagnosis, not retry permission. The supervisor must confirm the specific permitted failure, apply a correction, verify cleanup, and reserve another attempt. Architecture/schema/library errors, device loss, OOM, unknown state, and dispatched generation are not reclassified as asset retries.

Offline validation passed typecheck, all 76 JavaScript unit tests, worker/client builds, and 55 Python accounting/browser tests. The separate supervisor then completed both authorized arms using two loads, two dispatches, and 504 reported tokens, with verified cleanup. See the [v2 execution report](../reports/browser-integration-v2/README.md). This short check does not establish 4,096-token memory fit, sustained performance, or benchmark improvement.
