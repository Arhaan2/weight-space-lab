# Installed CLI control verification

This audit used the installed **Codex CLI 0.163.0** on 2026-11-06. It inspected
help/schema information and then tested the binary's configuration parser and
actual HTTP request serialization against a synthetic loopback server. The
[sanitized evidence](../configs/phase7_cli_control_evidence.json) records the
binary hash, settings, request fields, expected failures, and private diagnostic
harness hashes. No live provider, account credential, model generation, benchmark,
or experimental editor interaction was used.

The sanitized [reproduction harness](../scripts/verify_codex_controls.py) runs
the ten cases and writes only structural evidence to a new file:

```sh
.venv/bin/python scripts/verify_codex_controls.py --out <new-evidence-file>
```

`--case baseline --case ambiguous_drop_zero_retries` selects the two smoke
cases used to validate the public harness. Its source hash is recorded in the
evidence. It requires macOS and permission to install the loopback-only sandbox;
unsupported or refused sandbox launches fail closed. Each case has a local
deadline, bounded request/process-output bytes, and no live model invocation.
Temporary config/auth storage and captured streams are discarded; the configured
credential backend is file-only inside that empty temporary directory.

## Method and scope

The diagnostic launched the installed CLI with a fresh empty configuration home,
an empty working directory, `--ignore-user-config`, `--ignore-rules`,
`--strict-config`, `--ephemeral`, and an explicit custom Responses provider. That
provider pointed only at the local capture server, required no OpenAI auth, and
had zero request and stream retries. Analytics, feedback, apps, plugins, hooks,
browser/computer use, image generation, and daemon startup were disabled.
Captured requests had no Authorization header.

An OS Seatbelt profile denied outbound network except loopback. This diagnostic
profile is **not** the Phase 7 editor filesystem profile: it establishes that the
serialization check could not send a request to a remote provider, not that a
live editor was isolated. Raw prompts, tool dumps, temporary paths, and session
records were kept outside public content. Only structural evidence is retained.

The server returned HTTP 400 after capturing each request, deliberately stopping
before any inference. A separate case read the request and closed the connection
without a response to test ambiguous-outcome retry behavior. The diagnostic
subprocess also had a local wall-time bound; that bound is not a provider cap.

## Actual results

| Control/check | Behavior observed | What it establishes |
| --- | --- | --- |
| `codex exec --max-output-tokens 16 --help` | Exit 2: unexpected argument. | This CLI has no such flag. |
| Strict config `max_output_tokens=16` | Unknown configuration field; no request. | This override is unsupported. |
| Strict config `model_max_output_tokens=16` | Unknown configuration field; no request. | This override is unsupported. |
| Strict config `max_tokens=16` | Unknown configuration field; no request. | This override is unsupported. |
| Strict config `max_cost=1` or `max_billed_units=1` | Unknown configuration field; no request. | These overrides do not provide a provider billing ceiling. |
| `features.token_budget=true` | Accepted; actual request contains no `max_output_tokens` or `max_tokens`. | Acceptance does not establish a provider hard generation cap. |
| `features.rollout_budget={limit_tokens=16,reminder_at_remaining_tokens=[8]}` | Accepted; a `rollout_budget` marker is added to model-visible input; actual request contains no output-token cap. | This feature must not be substituted for a provider-enforced output limit. |
| Zero request/stream retries plus `features.unbounded_connection_retries=false` | Fake provider drops connection after reading the request; CLI exits 1 after exactly one request. | No automatic retry in this tested ambiguous local transport case. Live-provider behavior remains unverified. |
| Built-in tool feature switches | Some model-visible markers disappear, others remain; request `tool_choice` is `auto`. | A complete executable-tool allowlist has not been demonstrated. |

The actual request had the keys `client_metadata`, `include`, `input`, `model`,
`parallel_tool_calls`, `prompt_cache_key`, `reasoning`, `store`, `stream`, `text`,
and `tool_choice`. It had no provider hard output-token parameter. The absence
of a top-level `tools` array does not prove tool removal: the catalog declares a
code-mode tool interface, whose descriptions occur inside model-visible input.
Simple prompt-marker presence or absence is insufficient to authenticate the
executable tool registry, so this audit does not claim that tool removal worked.

These findings are scoped to the installed binary and tested controls. They do
not prove that every possible provider interface lacks hard caps. They do prove
that the tested CLI flags, configuration guesses, and accepted budget features
cannot establish the required provider-side frontier generation ceiling.

## Provenance levels

| Level | Evidence |
| --- | --- |
| Requested selector | `gpt-6-astra`, explicitly supplied. |
| Locally parsed/serialized selector | The installed CLI wrote `model: "gpt-6-astra"` to the synthetic local request. |
| Selector accepted by the real execution/provider interface | **Not obtained.** Local parsing/capture is not live acceptance. |
| Observed live session/response model metadata | **Not obtained.** The fake server supplied no model response. |
| Immutable model version | **Not exposed or verified.** CLI version is software provenance; the catalog's opaque `comp_hash` is not interpreted as a snapshot. |

No model self-description was solicited or used as identity evidence.
The experimental `allowProviderModelFallback` app-server field is documented by
the generated schema as controlling provider-catalog substitution, but its
effect was not verified against the real provider. It must remain an unverified
control until a suitable adapter exercises and records it.

## Consequence for Phase 7

The exact remaining transport capability is a demonstrated **provider-enforced
upper bound on frontier generation, including reasoning tokens**, with usage
and billing semantics appropriate to the declared budget. Reservations,
instructional budgets, usage notifications after execution, and killing the
local CLI cannot replace that control or prove that a remote request stopped.

The actual live editor launch must additionally use an enforced capability
boundary with a verified tool registry. The current tested Seatbelt worker runs
a Python fixture, while `EditorController.dispatch_live` has no live launch.
Consequently the tested fixture boundary cannot be claimed for a CLI launch.
The concrete retry configuration above is useful future adapter evidence, but
it does not remove these generation-limit and launch-isolation prerequisites.
