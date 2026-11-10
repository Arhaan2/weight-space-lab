# Astra execution-interface inspection

The interface inspection on **2026-11-06 (America/Los_Angeles)** found the
explicit selector **`gpt-6-astra`** in two actual local surfaces: this session's
`collaboration.spawn_agent` schema and the installed Codex CLI's bundled model
catalog. The catalog displays **`GPT-6-Astra`**. These are observed selector and
display strings, not an invented public API identifier. No experimental editor
request was sent, no live model identity or immutable model version was returned,
and no experiment ran. Live integration remains pending.

The machine-readable observation is
[`editor_interface_observation.json`](../configs/editor_interface_observation.json).
Null identity, version, endpoint, and billing fields mean **unavailable**, not
zero or an inferred value. The bundled catalog marks `supported_in_api: true`;
that local metadata does not establish an endpoint, account entitlement, or a
successful request. The CLI's software version is not the model's version.
The catalog also exposes `comp_hash: "3000"`, `multi_agent_version: "v2"`,
and `tool_mode: "code_mode_only"`. These exact values are recorded without
interpreting the opaque `comp_hash` as an immutable model build or the
multi-agent runtime version as the model's version.

## Sources actually inspected

No credentials, account history, or provider configuration were read. These
read-only commands inspected the installed runtime without sending a model
request:

```sh
codex --version
codex exec --help
codex sandbox --help
codex app-server --help
codex debug models --help
codex debug models --bundled
codex app-server generate-json-schema --out <temporary-directory>
codex app-server generate-json-schema --experimental --out <temporary-directory>
```

The executable reported **`codex-cli 0.163.0`**. The observation records its
SHA256 and hashes of the generated schemas used below. The full model catalog
contains runtime prompt templates; only the Astra catalog metadata was retained.
The tool schema supplied to this session is separate evidence and has no exposed
interface version.

## What the inspected interfaces support

| Surface | Observed capability | Limit of the evidence |
| --- | --- | --- |
| `collaboration.spawn_agent` | Explicit `gpt-6-astra` model override | No tool allowlist, filesystem/network boundary, or hard token/billing cap arguments; agents share the host filesystem and inherit available tools. |
| `codex exec` | `--model`, JSONL events, final-response schema, ephemeral operation, and shell sandbox selection | Generic shell sandbox names do not prove held-out read denial across every model-visible tool. No hard output-token or billing limit appears in its help. |
| App-server `ThreadStartResponse` | Returned `model` and `modelProvider` fields | These fields were inspected as schema, not received from a live run; no immutable model build field was found. |
| App-server model reroute notification | `fromModel`, `toModel`, and `reason` | A future adapter must fail on substitution and preserve the event; selecting a model alone does not prove it executed. |
| App-server token usage notification | Input, cached input, cache-write input, output, reasoning-output, and total token counts | Counts are usage observations, not a proved execution-time cap or monetary bill. No usage was produced by this inspection. |
| Experimental app-server thread start | `allowProviderModelFallback`, dynamic tool declarations, named permission profiles | A future adapter should explicitly prohibit provider fallback; additive dynamic tools do not prove removal of built-in tools. |

The generated `ThreadStartResponse` schema describes `disabledPluginIds` as a
saved list that **“does not yet filter plugin capabilities.”** It cannot serve
as the isolation mechanism. The inspected stable `TurnStartParams` schema has
no hard generated-token or monetary budget parameter. Its legacy read-only and
workspace-write policies expose write and network settings, not an audited
final-test read boundary. The experimental named-profile facility is a possible
integration surface requiring its own implementation and denial tests.

`ModelVerificationNotification` in this version describes the
`trustedAccessForCyber` marker; it is not a model-version attestation. No claim
about an immutable Astra build is derived from that notification.

## Dispatch decision

The current multi-agent tool is suitable for engineering collaboration, but is
not the experimental editor transport: inherited tools can read host files and
use connected services. A fresh chat cannot repair those permissions. The
observed local CLI/schema are integration leads, not permission to bypass the
controller with an unrestricted Codex subprocess.

Before enabling a real development-only interaction, a provider adapter must
establish all of the following through execution, not prompt instructions:

- Explicit Astra selection and returned model provenance, recording the version
  actually exposed or an explicit unavailable value; reject model substitution.
- An editor environment that cannot read final tasks, targets, generations,
  scores, audit details, controller private state, or unrelated host files, and
  cannot reach them through network, inherited descriptors, child processes,
  plugins, or shell tools.
- Reservation before dispatch, enforceable token/time/cost ceilings where billed
  usage is available, cancellation, and conservative recovery after interruption.
- A strictly bounded development tool contract; no training, gradients, optimizer
  operations, architecture edits, or evaluation edits.

The Phase 7 file handoff and non-scientific fixtures can be developed and verified
without that live connection. A hand-written or simulated response must retain
its simulation provenance; neither an imported model-name string nor successful
offline tests authenticate an Astra interaction.
