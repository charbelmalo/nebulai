# SessionSeer — real-time agent observability as a nebulai subapp

**Status:** built. M0–M5 are shipped; this file remains the design and its
justification, and [`SESSIONSEER-HANDOVER.md`](SESSIONSEER-HANDOVER.md) records
what the build actually does — including the places where it departed from what
is written here, and why.
**Scope:** a fourth nebulai front-end. Where `tokens` / `sae` / `neurons` map a
*model's* concept space, SessionSeer maps an *agent's* trajectory space. It
shares the viewer shell, the honesty guardrails, and the export contract; it
does **not** share the reduce → cluster → name back-end, because a trajectory is
already low-dimensional and ordered. It is not a fourth Plan under the Units
contract, and the cross-pipeline propagation rule does not apply to it.

---

## 0. Verdict on the intern report

The report is good. Its central architectural claims are correct, and its
central *discipline* — provenance labels on every value, `missing` never
silently becoming `0`, agent self-report never becoming `verified` — is exactly
the ethos `backend/validate.py` and the namer's `n_labeled == 0` rule already
enforce in this repo. Adopt that wholesale.

But it was written from documentation, not from the machine. I checked every
load-bearing claim against the binaries actually installed here:

| | version |
|---|---|
| `codex` | `codex-cli 0.144.6` (`~/.hermes/node/bin/codex`) |
| `claude` | `2.1.222` (`~/.local/share/claude/versions/2.1.222`) |
| `hermes` | `Hermes Agent v0.16.0 (2026.6.5)` |

Roughly 80% confirmed, several claims **understated**, and five material errors.
Sections 1–2 are the ledger. Build from those, not from the original report.

---

## 1. Confirmed — with the evidence

### 1.1 Codex app-server (confirmed, and understated)

`codex app-server` exists and generates a version-matched schema bundle, exactly
as claimed:

```bash
codex app-server generate-json-schema --out ./codexschema
```

That emits `ServerNotification.json`, `ServerRequest.json`, a v1/ and v2/ tree,
and a 547 KB combined `codex_app_server_protocol.schemas.json`. Parsing the
`oneOf` method enums gives **68 server notifications and 10 server→client
requests** for 0.144.6. The report's claimed coverage is all present and then
some:

- lifecycle — `thread/started`, `thread/status/changed`, `thread/closed`, `thread/compacted`
- turns — `turn/started`, `turn/completed`, `turn/diff/updated`, `turn/plan/updated`
- items — `item/started`, `item/completed`, `item/agentMessage/delta`, `item/commandExecution/outputDelta`, `item/fileChange/patchUpdated`, `item/mcpToolCall/progress`
- usage — `thread/tokenUsage/updated`
- approvals — `item/commandExecution/requestApproval`, `item/fileChange/requestApproval`, `item/permissions/requestApproval`, `item/tool/requestUserInput`
- routing — `model/rerouted`, `model/verification`, `model/safetyBuffering/updated`
- **`hook/started` / `hook/completed`** — Codex hooks are real *and* surface as first-class app-server events
- **`fs/changed`** — a native filesystem-change notification
- warnings — `warning`, `guardianWarning`, `deprecationNotice`, `configWarning`

Transports are richer than the report implies. `--listen` accepts
`stdio://` (default), `unix://`, `unix://PATH`, `ws://IP:PORT`, and `off`; there
is also an `app-server daemon` + `app-server proxy` pair and a top-level
`codex remote-control`. Hooks are enabled on this machine (`features.hooks =
true` in `~/.codex/config.toml`).

### 1.2 Claude Code hooks (confirmed — including the ones I doubted)

Extracted from the 2.1.222 binary. **19 events**, and the report's list was
substantially right:

```
PreToolUse  PostToolUse  PostToolUseFailure  UserPromptSubmit  Notification
Stop  StopFailure  SubagentStart  SubagentStop  PreCompact  PostCompact
SessionStart  SessionEnd  PermissionRequest  PermissionDenied
TaskCreated  TaskCompleted  CwdChanged  FileChanged
```

`PostToolUseFailure`, `StopFailure`, `PermissionDenied`, `TaskCreated`/
`TaskCompleted`, `CwdChanged` and `FileChanged` all exist. That is a strong
observed-mode surface — good enough that Claude observed mode is close to
managed mode for everything except token-level streaming.

### 1.3 Hermes state.db (confirmed, and richer than described)

`~/.hermes/state.db` — **688 MB**, WAL confirmed (`-shm`/`-wal` present),
**1,939 sessions / 56,861 messages** on this machine. The `sessions` table
already carries most of Tier 1 and Tier 3 natively:

```
model, model_config, system_prompt, parent_session_id, started_at, ended_at,
end_reason, message_count, tool_call_count, input_tokens, output_tokens,
cache_read_tokens, cache_write_tokens, reasoning_tokens, billing_provider,
billing_mode, estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
pricing_version, api_call_count, cwd, rewind_count, archived
```

`parent_session_id` is a self-referencing FK — delegation lineage is native.
`cost_source` + `pricing_version` mean Hermes already solved the
"preserve the pricing source" problem the report asks for.

**Gotcha the report got right for the wrong reason:** version-check before
querying — but the version is in the `schema_version` **table** (value `15`),
*not* `PRAGMA user_version`, which reads `0`. A reconciler keying on the pragma
will conclude "unversioned" on every install.

### 1.4 Hermes TUI gateway (confirmed)

Real, at `~/.hermes/hermes-agent/tui_gateway/` (`entry.py`, `server.py`,
`ws.py`, `transport.py`, `event_publisher.py`). Its method vocabulary confirms
every stream the report promised: `message.start/delta/complete`,
`tool.start/started/generating/complete`, `reasoning.delta`, `thinking.delta`,
`approval.request/respond`, `clarify.request/respond`, `secret.request/respond`,
`sudo.request/respond`, `session.create/branch/resume/interrupt/steer/undo/usage/history/status`,
`spawn_tree.list/load/save`, `subagent.tool`, `subagent.interrupt`,
`delegation.pause/status`, `rollback.diff/list/restore`, `status.update`.

### 1.5 The citation (confirmed)

arXiv 2607.06184 is real: *"What Resolve Rate Hides: Trajectory Structure
Diagnostics for Coding Agents."* It supports the thesis, and it supplies a
better taxonomy than the report's — see §2.5.

### 1.6 Headless modes (two structured, one not — corrected during M1a)

```
codex exec --json                                        # JSONL events
claude -p --output-format stream-json --include-partial-messages --verbose
hermes -z PROMPT                                         # final text ONLY — see below
hermes acp                                               # the structured Hermes path
```

**Correction to an earlier draft of this section**, found while writing the
adapters. I listed `hermes -z` alongside the other two as a structured headless
mode. It is not. Its own help text says it prints "ONLY the final response text
to stdout. No banner, no spinner, no tool previews, no session_id line." There
is no event stream to parse, and an adapter claiming to read tool calls off it
would be inventing them.

So Hermes's DRIVEN capture is deliberately thin — process lifecycle at
`DETERMINISTIC` fidelity, a final message, an exit code, and a `capture_gaps`
list naming everything it cannot see. Tokens come from a second, `RECONCILED`
pass over `state.db` (§1.3), joined by launch time + cwd because
`--pass-session-id` injects the id into the *system prompt*, not into stdout.
When that join is ambiguous — two Hermes runs from the same directory in the
same second — the reconciler reports `MISSING` rather than attributing real
token counts to the wrong run.

`hermes acp` is the structured live path and is verified working
(`hermes acp --check` → "Hermes ACP check OK"; the implementation is
`~/.hermes/hermes-agent/acp_adapter/`, which emits `tool_call` start/complete,
`agent_thought_chunk`, `agent_message_chunk` and `plan` session updates). It is
scheduled for M3 rather than M1 because it is a bidirectional JSON-RPC client,
not a line parser. Note that ACP carries **no token usage either**, so even the
structured path needs the `state.db` reconciler.

**A second correction, in the opposite direction: `codex exec --json` is
*thinner* than the Codex app-server, not richer.** Seven event kinds
(`thread.started`, `turn.started/completed/failed`,
`item.started/updated/completed`) against the app-server's 68 JSON-RPC
notifications. That inverts the usual assumption that DRIVEN — the mode where we
own the process — is the highest-fidelity mode. For Codex it is not, and the
adapter emits the gap list (`approval requests/decisions`, `per-request model
timing`, `context-window pressure`, `token usage before turn end`) at session
start so a chart can never quietly present a DRIVEN Codex run as if those were
observed and empty.

This matters more than the report realised — see §3.2.

---

## 2. Corrections — build against these, not the report

### 2.1 Hermes observed-mode is far stronger than claimed, and `gateway` is the wrong door

The report says Hermes observed mode "depends on gateway or persistence" and
routes integration through "the TUI gateway". Two errors.

**First:** `hermes gateway` is the **messaging** gateway — Telegram, Discord,
WhatsApp, Weixin. It is not the TUI gateway. The TUI gateway is spawned by
`hermes --tui` and has no top-level subcommand.

**Second:** Hermes has **three** hook systems, and the report noticed none of
them:

| System | Registered via | Runs in |
|---|---|---|
| Gateway hooks | `HOOK.yaml` + `handler.py` in `~/.hermes/hooks/` | gateway only |
| Plugin hooks | `ctx.register_hook()` in a plugin | **CLI + gateway** |
| Shell hooks | `hooks:` block in `~/.hermes/config.yaml` | **CLI + gateway** |

Shell-hook events (`hermes hooks test <event>`):

```
pre_tool_call   post_tool_call   pre_llm_call   post_llm_call
on_session_start  on_session_end  on_session_finalize  on_session_reset
subagent_stop   pre_gateway_dispatch
pre_approval_request   post_approval_response
transform_tool_result  transform_terminal_output  transform_llm_output
```

That maps almost one-to-one onto the canonical model. **Hermes observed mode is
a first-class path, not a fallback**, and it needs no TUI gateway at all.

Two rules follow. SessionSeer registers **only** the observe-only events and
**never** a `transform_*` hook — those mutate the agent's own data flow, and an
observability tool that can alter tool results is not an observability tool.
And consent is explicit: Hermes gates first use behind
`~/.hermes/shell-hooks-allowlist.json`, so installation is a two-step flow with
a real user approval, not a silent config write.

### 2.2 Do not write a Rust daemon

The report recommends a new cross-platform Rust daemon, `nebulai-agentd`. Reject
for v1.

This repo is Python + a TS/Preact viewer, and it **already has this exact
component**: `src/nebulai/backend/build_server.py` — an 18 KB stdlib HTTP
server that the viewer discovers via a settings URL with a health dot, and which
runs the real pipeline as a subprocess (`docs/LIVE-BUILDER.md`). SessionSeer's
collector is the same shape with a different payload. A Rust daemon adds a third
toolchain, a build step, and a packaging story to buy nothing v1 needs.

**But the report's latency budget is real and its own design misses it.** It
asks for a hook shim at p95 < 25 ms that writes to a Unix socket with a spool
fallback. A Python shim cannot start in 25 ms; interpreter boot alone is
30–50 ms. So invert it:

> **The hook shim is not a client. It is `>>` on a spool file.**
> Hooks append one JSON line to `~/.nebulai/spool/<agent>.jsonl`. The collector
> tails the spool. There is no socket in observed mode.

That is ~2–5 ms in `sh`, has no daemon dependency by construction, survives
collector restarts for free, and deletes the entire "spool fallback" branch and
its dedupe logic. Sockets are for *managed* adapters, where the collector is
already the parent process and the cost is amortised over a long-lived stream.

### 2.3 The whole Codex app-server is experimental, not just its WebSocket

The report says "the WebSocket transport is currently documented as experimental
and unsupported". `codex app-server --help` marks the **entire subcommand**
`[experimental]`, as it does `generate-ts` and `generate-json-schema`
individually. Consequence: pin the Codex version, ship the generated schema
bundle as a golden fixture per supported version, and **fail closed with a
visible compatibility banner** on an unrecognised version. Do not soft-degrade.

### 2.4 Codex *does* stream reasoning text — so opting out is a policy act

The report asserts hidden chain-of-thought "is not a valid observability target"
and "not available", and treats that as settled by capability. For Codex it is
not:

```
item/reasoning/textDelta
item/reasoning/summaryTextDelta
item/reasoning/summaryPartAdded
```

Hermes likewise emits `reasoning.delta` and `thinking.delta`, and its `messages`
table has `reasoning`, `reasoning_content`, `reasoning_details`, and
`codex_reasoning_items` columns.

So the data will arrive whether or not SessionSeer wants it. Not-storing-it is
an **active decision the code must implement**, not an absence it can rely on.
Concretely: the Codex adapter must drop `item/reasoning/textDelta` payloads at
the ingress boundary, before redaction, before the blob store — and the
data-quality panel must say `reasoning: dropped by policy`, never `missing`,
because "missing" would be a lie about why it isn't there.

### 2.5 The action taxonomy is too big and is missing the part that matters

The report proposes 18 normalized categories. The paper it cites uses **nine**
action types plus **deterministic effect labels** — and the effect label is the
load-bearing half. Repetition alone is not a loop; repetition *with no state
change* is. Without an effect label, §8.3's loop detector cannot work.

Adopt: 9 actions × an effect label.

```
action:  inspect · search · edit · execute · verify · vcs · delegate · interact · report
effect:  new_information | no_new_information | state_changed | no_state_change | failed
```

`verify` deliberately absorbs test/build/lint/typecheck — the research question
is "did verification happen", and splitting it four ways only creates four
sparse cells. Keep the native tool name alongside, always.

### 2.6 Minor, but fix them

- **`loop_score = a × b × c × d`.** Four normalized factors multiplied collapse
  to ~0 almost always, and the product is uninterpretable when it doesn't. Use
  a rule with a count and cited evidence — "4 equivalent searches, 0 new files"
  — not a score. This repo does not ship unfalsifiable numbers.
- **Claude's OTel intervals are tunable**, not fixed: `OTEL_LOGS_EXPORT_INTERVAL`
  and `OTEL_METRIC_EXPORT_INTERVAL` are both read from env. The report's
  conclusion (hooks for live, OTel for reconciliation) still holds — batch
  export is still batch — but not for the reason given.
- **state.db has no tool-span timing.** `messages` has one timestamp per row;
  there is no per-tool-call duration or exit-code table. The reconciler
  recovers turn-level and session-level facts only. The report's cross-agent
  matrix implies tool lifecycle is recoverable from persistence. It is not.
- **Network tracking: cut it.** Per-process attribution on macOS is coarse and
  privileged, and every genuinely useful web/MCP call already arrives natively.
  The report marks it optional; make it absent.
- **Live judge-model failure attribution: defer past v1.** Expensive,
  self-validating, and the 13-category taxonomy is unvalidated. Rule-based
  attribution plus human labels first.

### 2.7 The existing Sessions tab already does part of this

`viewer/src/chrome/sessionlog.ts` (591 lines) already parses Claude transcripts
into turns, and it already learned the single most important lesson in this
whole problem:

> Claude Code writes one model response as **several** JSONL lines, repeating
> identical `usage` on each. Counting per-line overcounts tokens and turns
> **3.5×** on a real session. The honest unit is the `requestId`.

There is also `sessionStore.ts` (IndexedDB persistence, derived-only), a
`SessionFieldDriver` (three/webgpu + TSL, 1,470 lines), and the audit-format
`result`-line ground-truth path with an `outputReliable` flag.

SessionSeer **extends this**, it does not greenfield next to it. And the
requestId lesson generalises into a canonical-contract rule: *the event stream's
line granularity is never the analysis granularity.* Codex has the same hazard
(`item/*/delta` vs `item/completed`) and so does Hermes (`message.delta` vs
`message.complete`). Fold on the native completion event, count usage once.

### 2.8 Merge-not-clobber is not hypothetical here

`~/.claude/settings.json` on this machine already has a live
`PreToolUse`/`Bash` hook running `rtk hook claude`. An installer that writes
rather than merges silently disables RTK's token savings. Back up, merge,
verify, and provide exact restoration — as the report says, but treat it as a
tested requirement, not a nicety.

---

## 3. What SessionSeer is

### 3.1 The six questions (kept from the report)

What was tested · what the agent did · where time/tokens/attention went · where
the trajectory turned · what evidence supports the outcome · can it be
reproduced and fairly compared.

### 3.2 The differentiator: promote cross-agent comparison to v1

The report puts run comparison in Milestone 6 of 7. That is backwards. Three
facts make it the *cheapest* high-value thing here:

1. All three agents have a working headless mode (§1.6).
2. All three headless modes are **managed** capture — highest fidelity, no
   installer, no config mutation, no consent flow, nothing to uninstall.
3. nebulai already has a Compare view with per-state normalization and a
   4-state layout transport.

So v1's spine is:

> **One task → three agents → three trajectories → one honest diff.**

That is a real instrument for an LLM research scientist on day one. Passive
observation of interactive sessions is *additive* to it, not a prerequisite.

**And the honest-comparison rule is the product.** Claude's `cache_read`,
Codex's `thread/tokenUsage`, and Hermes's `cache_read_tokens` +
`reasoning_tokens` are not the same quantity. Extend the `validate.py` ethos: a
**comparability gate** that refuses to render a cross-agent token or cost delta
when the native categories don't align, and says why. Aligned axes (wall-clock,
action counts, files touched, verification presence, approvals) compare freely.
Unaligned ones show side-by-side with native labels and no delta.

### 3.3 Capture ladder (renamed for accuracy)

| Mode | How | Fidelity |
|---|---|---|
| **Driven** | SessionSeer launches the agent headless and owns stdout | full, no install |
| **Attached** | SessionSeer connects to a running app-server / TUI gateway | full, needs a live endpoint |
| **Observed** | Hooks append to the spool; SessionSeer tails it | strong lifecycle, no token stream |
| **Reconciled** | state.db / transcripts / git, after the fact | totals and history only |

Every value carries `native | deterministic | estimated | heuristic | missing |
dropped_by_policy`. The last one is new and load-bearing (§2.4).

---

## 4. Architecture

```
codex exec --json ─────────┐
claude -p stream-json ─────┤  DRIVEN (subprocess, stdout)
hermes -z / acp ───────────┘
                           │
codex app-server ──────────┤  ATTACHED (stdio / unix / ws)
hermes tui_gateway ────────┘
                           │
~/.nebulai/spool/*.jsonl ──┤  OBSERVED (hooks append; collector tails)
                           │
~/.hermes/state.db (ro) ───┤  RECONCILED
~/.claude/projects (opt-in)┤
~/.codex/sessions (opt-in) ┤
git + fs snapshots ────────┘
                           ▼
              nebulai.backend.seer_server   (stdlib HTTP + SSE, port 8125)
                ingress → policy/redaction → append-only JSONL
                        → reducer → SQLite index → derived metrics
                           ▼
              GET /seer/live (SSE) · /seer/runs · /seer/export
                           ▼
                  viewer: SessionSeer page (new nav pill)
```

Port **8125**, deliberately beside the build server's 8124, same discovery
pattern, same health dot.

**Storage.** Canonical log is append-only JSONL under `~/.nebulai/seer/` — the
same "immutable artifact, regenerable derivations" split the map pipeline
already uses. SQLite indexes it. Large payloads go to a content-addressed blob
dir referenced by hash. Parquet/DuckDB is an export target, not a runtime
dependency. The viewer keeps IndexedDB for the derived analyses it already
persists, so SessionSeer works with the server *off* for anything already
loaded.

---

## 5. Data contract

Lock this before any adapter. It is the artifact that stops three dashboards
from happening.

```jsonc
{
  "schema_version": "1.0",
  "event_id": "evt_...", "ts": "2026-08-05T12:34:56.789Z", "mono_ns": 8273649287364,

  "source": {
    "agent": "codex|claude|hermes",
    "agent_version": "0.144.6",
    "adapter": "codex_app_server", "adapter_version": "0.1.0",
    "capture_mode": "driven|attached|observed|reconciled",
    "fidelity": "native|deterministic|estimated|heuristic|missing|dropped_by_policy",
    "source_event_id": "item_123"
  },

  "run_id": "run_...", "session_id": "ses_...", "turn_id": "turn_...",
  "span_id": "span_...", "parent_span_id": "span_...",

  "event_type": "tool.completed",

  "repo":  { "root_id": "sha256:…", "branch": "…", "head": "…", "dirty": true },
  "model": { "provider": "…", "model_id": "…", "effort": "high" },

  "action": "execute",              // 9-type taxonomy
  "effect": "state_changed",        // deterministic effect label
  "native_type": "commandExecution",

  "payload": { "status": "completed", "duration_ms": 18234, "exit_code": 0 },
  "privacy": { "content_level": "redacted", "ruleset": "r7" }
}
```

Event families follow the report's list, minus the ones nothing emits. Two
additions from §2.7: every family has a designated **fold key** (the native
completion event) and a rule that deltas never contribute to counters.

State machine as the report specifies, with `stalled` and `overdue` as
**overlays** on the underlying state, not replacements. That was one of its
better calls.

---

## 6. Analyses that survive scrutiny

Keep, with the evidence panel behind each:

- **Time decomposition** — inclusive / exclusive / concurrency-weighted, never
  summed across overlapping subagent spans.
- **Verification coverage** — changed files vs. observed `verify` actions after
  the last edit. Rule-based, per project type, evidence listed.
- **Edit churn** — `churn_ratio = cumulative_lines / max(final_lines, 1)`.
- **Loop detection** — repetition **and** `effect == no_new_information`, as a
  counted rule with cited events (§2.5, §2.6). No product score.
- **Human intervention burden** — approvals, corrections, wait time.
- **Context pressure** — compaction count, before/after tokens where reported.
- **Progress evidence** — the checklist, never a percentage.

Defer: live judge-model failure attribution; the 13-category taxonomy;
"cost to verified outcome" until evaluators are real.

Outcome states as specified — `agent_claimed_complete` and `verified_pass` must
never collapse. That is the same rule as "a namer with `n_labeled == 0` says so".

---

## 7. UI

A new nav pill beside **Sessions** in `viewer/src/chrome/TopBar.tsx`, a `Page`
variant, a `body.page-seer` class in `mount.tsx`, and a `SeerPage.tsx`. Existing
Sessions stays as the drop-a-transcript forensic view; SessionSeer is the live
and comparative one.

Three panes:

1. **Live** — one card per active session: agent/model, repo/branch/commit,
   state + current action + elapsed, context/tokens/cost where native, files and
   diff summary, latest verification, pending approval, subagent count, alerts,
   and a capture-health chip.
2. **Trajectory** — synchronized lanes (prompts, model, tools, files/git,
   verification, approvals, subagents, alerts). Reuses `SessionFieldDriver`.
3. **Compare** — N runs aligned by wall-clock / turn / action sequence / first
   edit / first verification, with the §3.2 comparability gate enforced.

Plus the report's **data-quality panel** on every session. Non-negotiable: the
map viewer already distinguishes "no data" from "zero" everywhere, and a
comparison without a completeness readout is not a scientific object.

---

## 8. Privacy

Three tiers as specified — metadata-only (default), redacted research, full
local (explicit opt-in). Redaction before persistence, `.nebulaiignore`, blobs
encrypted at rest with the key from the OS keychain, loopback-only ingress,
per-run and full deletion, retention by data class.

Rejected capture methods stand as written: no keylogging, screen recording,
screenshots, terminal OCR, clipboard capture, packet inspection, browser
history, mic or camera.

Two additions:

- **Reasoning is dropped at ingress by policy** (§2.4), and labelled
  `dropped_by_policy`.
- **Observe and control are separate permissions.** All three agents expose
  approval *response* channels (`approval.respond`, Codex's approval requests,
  Claude's permission hooks). SessionSeer ships v1 with the responder
  unimplemented, not merely disabled.

---

## 9. Milestones

Tighter than the report's eight. Cross-agent comparison moves from M6 to M1.

**M0 — Contract.** Event envelope, 9×effect taxonomy, fidelity enum, outcome
states, privacy tiers, fold-key rule. Recorded fixtures from all three agents.
*Exit:* one synthetic run representable with zero agent-specific fields leaking
into the canonical schema; every metric names its required events and its
fallback.

**M1 — Driven triple + comparison.** `seer_server` on 8125; driven adapters for
`codex exec --json`, `claude -p --output-format stream-json`, `hermes -z`;
JSONL store; reducer; SSE; SeerPage Live + Compare; comparability gate.
*Exit:* one task run on all three agents, three trajectories, one honest diff,
and every unaligned metric visibly refusing to subtract.

**M2 — Observed mode.** Spool-file hook shim; installers for Claude (19 events),
Codex, and Hermes shell hooks — each merging, backing up, and restoring exactly,
verified against the live `rtk hook claude` entry. Consent flow for Hermes's
allowlist.
*Exit:* hooks add < 5 ms p95; agent sessions unaffected when the server is down;
no existing hook disturbed.

**M3 — Attached mode + reconciliation.** Codex app-server (version-pinned,
schema bundle as golden fixture, fail-closed on unknown versions); Hermes TUI
gateway; read-only `state.db` reconciler keyed on the `schema_version` **table**;
git/fs snapshots.
*Exit:* reconnect without duplicates; active and persisted sessions never
double-counted; unknown schema versions disable the importer rather than guess.

**M4 — Analyses + export.** Time decomposition, verification coverage, churn,
loop rules, intervention burden, context pressure, progress evidence,
annotations, JSONL/Parquet export.
*Exit:* every derived value deterministic from the log and carrying a version,
formula, inputs, and evidence.

**M5 — Hardening.** Golden fixtures per supported agent version, compatibility
CI, redaction test suite, crash/reconnect, installer/uninstaller validation,
deletion tests.
*Exit:* upgrading an agent cannot silently corrupt historical metrics;
unsupported versions warn visibly; uninstall restores byte-exact config.

> As built, "restores byte-exact config" was replaced by three properties that
> are separately true and separately testable — the *backup* is the byte-exact
> undo, a text config round-trips byte-exactly, and a JSON config round-trips
> semantically. Uninstall removes by tag rather than restoring, so hooks the
> user added after installing survive it. See the handover, §4.

---

## 10. Risks

Inherit the report's table. The three that actually bite here:

- **Agent APIs move fast, and Codex's is flagged experimental in full.** Version
  pinning plus generated schema bundles plus fail-closed is the whole mitigation.
  Accept that a Codex upgrade can disable the attached adapter until refreshed.
- **Cross-agent token/cost semantics genuinely do not align.** The comparability
  gate is not defensive polish; without it the headline number is wrong and
  looks authoritative. This is the same failure mode as the neuron map's null
  model clearing by only +0.056 — a plausible number that means less than it
  appears.
- **Installer clobbering live user config.** Concrete and present (§2.8).
  Merge, back up, verify, restore — and test it.

---

## Appendix — reproducing the validation

```bash
codex --version && claude --version && hermes --version
codex app-server generate-json-schema --out /tmp/codexschema
strings -n 4 ~/.local/share/claude/versions/2.1.222 | grep -oE '"(PreToolUse|PostToolUse|PostToolUseFailure|UserPromptSubmit|Stop|StopFailure|SubagentStart|SubagentStop|PreCompact|PostCompact|SessionStart|SessionEnd|PermissionRequest|PermissionDenied|TaskCreated|TaskCompleted|CwdChanged|FileChanged|Notification)"' | sort -u
hermes hooks list && hermes hooks test --help
sqlite3 ~/.hermes/state.db "SELECT * FROM schema_version;" && sqlite3 ~/.hermes/state.db ".schema sessions"
```
