# Authenticated Astra session boundary audit

The existing subscription session is the requested authentication route. The
remaining blocker is an **enforced and verified experimental-session access
boundary**. The desktop chat-creation interface exposes Astra selection but
does not expose the permissions needed to establish that boundary. No fresh
experimental session, model generation, weight proposal, candidate checkpoint,
or development comparison was performed by this audit.

The sanitized observation is
[`session_interface_observation.json`](../configs/session_interface_observation.json).
The inspection was performed on 2026-11-06 using supplied desktop tool
declarations and local **Codex CLI 0.163.0** help and generated experimental
app-server schemas. No credentials, account conversations, held-out assets, or
global configuration were read or changed. Schema generation did not start an
app server or model. Raw schemas and machine-specific paths are not published.

## Actual session controls

| Surface | Observed capability | Missing evidence or limitation |
| --- | --- | --- |
| Desktop `create_thread` | Fresh chat, explicit `model`, explicit `thinking`, project or projectless destination. It exposes `gpt-6-astra` and efforts `low`, `medium`, `high`, `xhigh`, `max`, `ultra`. | No per-session permission profile, filesystem read boundary, network allowlist, executable-tool allowlist, plugin filter, or retry policy arguments. Creation dispatches an initial prompt; it is not an inert session whose boundary can be verified before generation. |
| Desktop chat management | Read, follow up, wait, archive, and organize chats. | No inspected desktop tool supplies a per-chat capability-boundary setter. Projectless and worktree choices do not demonstrate denied access to the host, harness, graders, or connectors. |
| Experimental app-server `ThreadStartParams` | Named `permissions`, `dynamicTools`, `allowProviderModelFallback`, `environments`, `runtimeWorkspaceRoots`, and `selectedCapabilityRoots`. | These are schema fields, not tested enforcement. They are not arguments of the exposed desktop `create_thread` tool. Dynamic tools do not establish removal of built-ins. A custom app-server transport would be additional integration work, not proof that the current desktop session is restricted. |
| Experimental app-server response | `activePermissionProfile`, `model`, `modelProvider`, and `reasoningEffort`. | No live response was obtained. A named profile identifies configuration provenance; denial tests must establish its effective behavior. |
| App-server `disabledPluginIds` | A persisted disabled-plugin list. | Its description explicitly says it **does not yet filter plugin capabilities**. It cannot establish isolation. |
| `codex exec --sandbox` | Shell-command sandbox selection. | Its documented scope does not authenticate a complete model-visible tool boundary. A CLI inference process is also not the requested desktop session workflow. |
| `codex sandbox` | Named permission profile and opaque sandbox-state inputs; readable-root and network controls. | Wrapping a command does not establish that a desktop experimental session and every exposed tool use that wrapper. |
| Plugin permission management | Global/default and individual-plugin approval settings. | These are not per-session filesystem or capability restrictions; global mutations were not made. |

Native UI inspection was also unavailable: `cua.getApp('Codex')` was rejected
with **"Computer Use is not allowed to use the app com.openai.codex for safety
reasons."** No UI was inspected or changed, and no alternate UI access was
attempted. That capability restriction leaves no verified UI route for
configuring the fresh session's access boundary.

The app-server schema describes `environments: []` as disabling environment
access for turns without an override. That statement alone does not establish
that all filesystem, shell, network, app, browser, and resource-reading tools
are absent. Likewise, selected capability roots and runtime workspace roots
must not be treated as a deny-by-default tool registry without executable
verification.

The existing Phase 7 isolation test launches a Python fixture through the
controller's Seatbelt boundary. Its denial results apply to that launch.
They cannot be transferred to a desktop Astra chat by assertion. The current
engineering chat and collaboration agents retain tools capable of accessing
host files and connected services and therefore are not experimental editors.

## Model and usage provenance

| Provenance level | Evidence |
| --- | --- |
| Requested selector | `gpt-6-astra`, the user's requested Astra role mapped to the selector explicitly exposed by the desktop tool. |
| Selector accepted by live execution | Unavailable; no experimental session was launched. |
| Observed session/response model metadata | Unavailable; no live session/response evidence was obtained. |
| Immutable model version | Unavailable in the inspected evidence. The CLI version identifies software, not an Astra snapshot. |
| Effort | The bounded development attempt requests `high`. Supported values are exposed by the tool; accepted and observed effort remain unavailable. |

No model self-description is identity evidence. Local selector exposure does
not prove account entitlement, provider acceptance, or the model that executed.
The session variant must record returned metadata when available and must not
substitute another model to obtain a proposal.

The [installed-control audit](phase7-cli-controls.md) establishes that the
audited subscription route does not expose a demonstrated provider-enforced
frontier generation-token ceiling, including reasoning tokens. This is an
explicit limitation of the session variant, not a claim that ledger reservations
or local elapsed-time cancellation provide that ceiling. The strict automated
mode retains its own hard-cap requirements.

The desktop `get_usage_limits` tool exposes account-shared consumption
percentages and reset windows. Those are not per-request token usage or billed
charges. App-server usage-notification schemas include input, cached input,
cache-write input, output, reasoning-output, and total tokens; none were
observed live here. Unexposed fields remain null, not zero. Provider generation
and usage remain separate from locally enforced proposal/request counts,
target-model forward sequences/tokens, and local elapsed time.

## Smallest remaining prerequisite

A fresh session must be created through a host interface that can enforce a
development-only capability boundary **before the editor request is exposed**.
It must deny final tasks, answers, generations, scores, evaluator state,
unrelated host files, and harness/grader modification through every tool and
connector. The actual session must pass denial tests and expose enough runtime
evidence to tie that boundary and tool inventory to its launch. A no-tools
session could use the existing file handoff if the host proves tools and other
information channels are disabled; an instruction to avoid them is insufficient.

Neither the inspected desktop interface nor the existing fixture test provides
that proof. Adding an unverified permission field to a record, creating a fresh
chat with a restrictive prompt, or importing a purported Astra response would
not repair it. The smallest next integration capability is a desktop session
launch that accepts and attests the restricted permissions/tool set, followed
by actual denial checks on that session. The existing handoff, proposal
validator, checkpoint editor, and bounded development-probe machinery can then
consume its response without requiring a separate API key or nested inference.

This audit does not weaken the held-out-data boundary or scientific
restrictions, change the strict automated readiness gates, or authorize a
broader search or final benchmark evaluation.
