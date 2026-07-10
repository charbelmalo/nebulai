/** Sessions — a 3-D plotter for real agent-mode session transcripts. Drop a
 *  Claude Code `.jsonl` transcript and the session becomes an honest flight
 *  path through (time × context × output-per-turn), one node per model
 *  response (requestId-folded, so tokens are never over-counted). Overlay
 *  several to compare their shapes. Analyses persist to IndexedDB and rehydrate
 *  on load — the "analysis persistence from session to session" contract.
 *
 *  This supersedes the keyword Snapshot Map for large sessions: it keeps the
 *  token accounting, tool sequence, task lifecycle, and file touches the
 *  keyword map throws away. Everything runs client-side; raw transcript text is
 *  parsed in memory and never stored or transmitted. */

import { signal, useSignal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";
import { appStore } from "../app/store";
import {
  CATEGORY_ORDER,
  parseSessionTranscript,
  type SessionAnalysis,
  type ToolCategory,
} from "./sessionlog";
import { SessionPlotDriver, CATEGORY_RGB } from "../scene/sessions/SessionPlotDriver";
import {
  deleteSessionAnalysis,
  loadAllSessionAnalyses,
  saveSessionAnalysis,
} from "./sessionStore";
import { $sessions } from "./state";

/** Pinned turn = the node last clicked in the plot. Page-level (not persisted):
 *  the inspector, the plot highlight, and keyboard nav all read/write it. */
const $pinned = signal<{ sessionId: string; index: number } | null>(null);
/** Categories currently dimmed in the plot (legend-key toggles). */
const $dimmedCats = signal<ToolCategory[]>([]);

const CATEGORY_HELP: Record<ToolCategory, string> = {
  orient: "read / search / fetch",
  plan: "task lifecycle, plan mode",
  edit: "write / edit files",
  exec: "run commands",
  deliver: "present / publish / notify",
  reflect: "pure text or thinking",
};

const SAMPLE_TRANSCRIPT = [
  `{"type":"user","timestamp":"2026-07-09T18:00:00Z","message":{"role":"user","content":"Refactor the auth module and add tests."}}`,
  `{"type":"assistant","requestId":"req-a","timestamp":"2026-07-09T18:00:12Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":40,"output_tokens":180,"cache_read_input_tokens":12000,"cache_creation_input_tokens":8000},"content":[{"type":"thinking","thinking":"look at the module"}]}}`,
  `{"type":"assistant","requestId":"req-a","timestamp":"2026-07-09T18:00:12Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":40,"output_tokens":180,"cache_read_input_tokens":12000,"cache_creation_input_tokens":8000},"content":[{"type":"tool_use","name":"Read","input":{"file_path":"src/auth.ts"}}]}}`,
  `{"type":"user","timestamp":"2026-07-09T18:00:20Z","message":{"role":"user","content":[{"type":"tool_result","content":"...file contents..."}]}}`,
  `{"type":"assistant","requestId":"req-b","timestamp":"2026-07-09T18:00:40Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":30,"output_tokens":900,"cache_read_input_tokens":22000,"cache_creation_input_tokens":3000},"content":[{"type":"tool_use","name":"Edit","input":{"file_path":"src/auth.ts"}}]}}`,
  `{"type":"assistant","requestId":"req-c","timestamp":"2026-07-09T18:01:20Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":20,"output_tokens":520,"cache_read_input_tokens":30000,"cache_creation_input_tokens":1500},"content":[{"type":"tool_use","name":"Bash","input":{"command":"npm test"}}]}}`,
  `{"type":"assistant","requestId":"req-d","timestamp":"2026-07-09T18:01:50Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":10,"output_tokens":260,"cache_read_input_tokens":33000,"cache_creation_input_tokens":600},"content":[{"type":"text","text":"Done — auth refactored and tests pass."}]}}`,
].join("\n");

export function SessionsPage() {
  const sess = $sessions.value;
  const active = sess.analyses.filter((a) => sess.activeIds.includes(a.id));
  const dragOver = useSignal(false);

  // ── one-time rehydrate from IndexedDB ──────────────────────────────────
  // MERGE (not replace): a session parsed during the async read must survive,
  // so anything already in memory wins over a same-id persisted copy.
  useEffect(() => {
    if (appStore.getState().sessions.hydrated) return;
    loadAllSessionAnalyses()
      .then((list) => {
        const cur = appStore.getState().sessions.analyses;
        const have = new Set(cur.map((a) => a.id));
        const merged = [...cur, ...list.filter((a) => !have.has(a.id))];
        appStore.getState().setSessionAnalyses(merged);
      })
      .catch(() => appStore.getState().setSessionsHydrated(true));
  }, []);

  return (
    <div class="sessions-page" role="main">
      <div class="sessions-shell">
        <SessionsSide dragOver={dragOver.value} setDragOver={(v) => (dragOver.value = v)} />
        <div class="sessions-stage">
          <SessionPlot analyses={active} />
          <SessionsLegend />
          <TurnInspector analyses={active} />
          {active.length === 0 && (
            <div class="sessions-empty">
              <h2>Plot an agent session</h2>
              <p>
                Drop a Claude&nbsp;Code <code>.jsonl</code> transcript (or load the sample) to see
                the session as a flight path through <b>time × context × output</b>. Overlay several
                to compare their shape.
              </p>
              <button type="button" class="btn-primary" onClick={() => loadSample()}>
                Load sample session
              </button>
            </div>
          )}
          {active.length > 0 && <SessionsStats sessions={active} />}
        </div>
      </div>
    </div>
  );
}

// ── the 3-D canvas host ───────────────────────────────────────────────────

function SessionPlot(props: { analyses: SessionAnalysis[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const driverRef = useRef<SessionPlotDriver | null>(null);
  const ready = useSignal(false);

  // init once
  useEffect(() => {
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    const host = canvas?.parentElement;
    if (!canvas || !overlay || !host) return;
    let disposed = false;
    const driver = new SessionPlotDriver();
    driver.onSelect = (sel) => {
      $pinned.value = sel;
    };
    driver
      .init(canvas, overlay)
      .then(() => {
        if (disposed) {
          driver.dispose();
          return;
        }
        driverRef.current = driver;
        const r = host.getBoundingClientRect();
        driver.resize(r.width, r.height, window.devicePixelRatio || 1);
        ready.value = true;
      })
      .catch((e) => console.error("session plot init failed", e));

    const ro = new ResizeObserver(() => {
      const r = host.getBoundingClientRect();
      driverRef.current?.resize(r.width, r.height, window.devicePixelRatio || 1);
    });
    ro.observe(host);

    return () => {
      disposed = true;
      ro.disconnect();
      driverRef.current?.dispose();
      driverRef.current = null;
    };
  }, []);

  // push data whenever the active set changes
  const key = props.analyses.map((a) => a.id).join("|");
  useEffect(() => {
    if (!ready.value) return;
    driverRef.current?.setSessions(props.analyses);
    // drop a pinned turn whose session left the active set
    const p = $pinned.value;
    if (p && !props.analyses.some((a) => a.id === p.sessionId)) $pinned.value = null;
  }, [key, ready.value]);

  // mirror page state → driver (selection ring, category dimming)
  const pinned = $pinned.value;
  const dimmed = $dimmedCats.value;
  useEffect(() => {
    if (ready.value) driverRef.current?.setSelected(pinned);
  }, [pinned, ready.value]);
  useEffect(() => {
    if (ready.value) driverRef.current?.setCategoryFilter(dimmed);
  }, [dimmed, ready.value]);

  return (
    <div class="sessions-canvas-host">
      <canvas ref={canvasRef} class="sessions-canvas" />
      <div ref={overlayRef} class="sessions-overlay" />
    </div>
  );
}

// ── left rail: ingest + session list ──────────────────────────────────────

function SessionsSide(props: { dragOver: boolean; setDragOver: (v: boolean) => void }) {
  const sess = $sessions.value;
  const pasteText = useSignal("");
  const errorMsg = useSignal("");
  const fileRef = useRef<HTMLInputElement>(null);

  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    props.setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) doParse(await file.text(), file.name, errorMsg);
  };

  return (
    <aside class="sessions-side">
      <section
        class={`sessions-side-block sessions-drop${props.dragOver ? " is-drag" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          props.setDragOver(true);
        }}
        onDragLeave={() => props.setDragOver(false)}
        onDrop={onDrop}
      >
        <h3>Load session</h3>
        <p class="sessions-hint">
          Claude&nbsp;Code transcript <code>.jsonl</code>. Parsed locally — never uploaded.
        </p>
        <input
          ref={fileRef}
          type="file"
          accept=".jsonl,.json,application/json,text/plain"
          class="visually-hidden"
          onChange={async (e) => {
            const f = (e.currentTarget as HTMLInputElement).files?.[0];
            if (f) doParse(await f.text(), f.name, errorMsg);
          }}
        />
        <div class="sessions-drop-actions">
          <button type="button" class="btn-ghost" onClick={() => fileRef.current?.click()}>
            Choose file
          </button>
          <button type="button" class="btn-ghost" onClick={() => loadSample()}>
            Sample
          </button>
        </div>
        <textarea
          class="sessions-paste"
          placeholder="…or paste JSONL"
          rows={4}
          value={pasteText.value}
          onInput={(e) => (pasteText.value = (e.currentTarget as HTMLTextAreaElement).value)}
        />
        <button
          type="button"
          class="btn-primary"
          disabled={!pasteText.value.trim()}
          onClick={() => {
            doParse(pasteText.value, `pasted-${sess.analyses.length + 1}`, errorMsg);
            pasteText.value = "";
          }}
        >
          Parse
        </button>
        {errorMsg.value && <p class="sessions-error">{errorMsg.value}</p>}
      </section>

      {sess.analyses.length > 0 && (
        <section class="sessions-side-block">
          <h3>
            Sessions <span class="sessions-count">{sess.analyses.length}</span>
          </h3>
          <ul class="sessions-list">
            {sess.analyses.map((a, i) => {
              const on = sess.activeIds.includes(a.id);
              return (
                <li key={a.id} class={on ? "is-active" : ""}>
                  <button
                    type="button"
                    class="sessions-item"
                    aria-pressed={on}
                    onClick={() => appStore.getState().toggleSessionActive(a.id)}
                  >
                    <span class="sessions-swatch" style={swatchStyle(i)} />
                    <span class="sessions-item-body">
                      <span class="sessions-item-name">{a.name}</span>
                      <span class="sessions-item-meta">
                        {a.nAssistant} resp · {fmtTok(a.totalOutput)} out ·{" "}
                        {a.spanSec >= 90 ? `${(a.spanSec / 60).toFixed(1)}m` : `${Math.round(a.spanSec)}s`}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    class="sessions-item-x"
                    aria-label={`Remove ${a.name}`}
                    onClick={() => {
                      appStore.getState().removeSessionAnalysis(a.id);
                      deleteSessionAnalysis(a.id).catch(() => {});
                    }}
                  >
                    ×
                  </button>
                </li>
              );
            })}
          </ul>
          <p class="sessions-persist-note">Saved locally · restored next visit</p>
        </section>
      )}
    </aside>
  );
}

// ── legend + stats ─────────────────────────────────────────────────────────

function SessionsLegend() {
  const dimmed = $dimmedCats.value;
  return (
    <div class="sessions-legend">
      <div class="sessions-legend-title">
        turn category <span class="sessions-legend-hint">click to dim</span>
      </div>
      <ul class="sessions-legend-keys">
        {CATEGORY_ORDER.map((c) => {
          const off = dimmed.includes(c);
          return (
            <li key={c}>
              <button
                type="button"
                class={`sessions-key${off ? " is-off" : ""}`}
                aria-pressed={off}
                title={off ? `show ${c} turns` : `dim ${c} turns`}
                onClick={() => {
                  $dimmedCats.value = off
                    ? dimmed.filter((x) => x !== c)
                    : [...dimmed, c];
                }}
              >
                <span
                  class="sessions-key-dot"
                  style={{ background: `rgb(${CATEGORY_RGB[c].join(",")})` }}
                />
                <span class="sessions-key-label">{c}</span>
                <span class="sessions-key-help">{CATEGORY_HELP[c]}</span>
              </button>
            </li>
          );
        })}
      </ul>
      <div class="sessions-legend-axes">
        X time · Y context (cache-read) · Z new-context/turn (cache-write) · size ∝ tools ·
        faded = sub-agent · drag to orbit · <b>click a node to dissect it</b>
      </div>
    </div>
  );
}

// ── turn inspector — click a node, dissect the turn ────────────────────────

function TurnInspector(props: { analyses: SessionAnalysis[] }) {
  const pin = $pinned.value;
  const a = pin ? props.analyses.find((x) => x.id === pin.sessionId) : undefined;
  const t = pin && a ? a.turns[pin.index] : undefined;

  // ← / → step through the session's responses; Esc unpins. Skipped while the
  // user is typing in an input (the paste box).
  useEffect(() => {
    if (!pin || !a || !t) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      if (e.key === "Escape") $pinned.value = null;
      else if (e.key === "ArrowLeft" && pin.index > 0)
        $pinned.value = { sessionId: pin.sessionId, index: pin.index - 1 };
      else if (e.key === "ArrowRight" && pin.index < a.turns.length - 1)
        $pinned.value = { sessionId: pin.sessionId, index: pin.index + 1 };
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pin?.sessionId, pin?.index, a?.turns.length]);

  if (!pin || !a || !t) return null;

  const files = [...new Set(t.files)];
  return (
    <div class="sessions-inspect" role="dialog" aria-label={`Response ${t.index + 1} of ${a.name}`}>
      <div class="sessions-inspect-head">
        <div class="sessions-inspect-title">
          <span class="sessions-inspect-name">{a.name}</span>
          <span class="sessions-inspect-turn">
            response {t.index + 1} / {a.turns.length}
            {t.isSidechain ? " · sub-agent" : ""}
          </span>
        </div>
        <div class="sessions-inspect-nav">
          <button
            type="button"
            class="sessions-inspect-btn"
            aria-label="Previous response"
            disabled={t.index === 0}
            onClick={() => ($pinned.value = { sessionId: a.id, index: t.index - 1 })}
          >
            ←
          </button>
          <button
            type="button"
            class="sessions-inspect-btn"
            aria-label="Next response"
            disabled={t.index >= a.turns.length - 1}
            onClick={() => ($pinned.value = { sessionId: a.id, index: t.index + 1 })}
          >
            →
          </button>
          <button
            type="button"
            class="sessions-inspect-btn"
            aria-label="Close inspector"
            onClick={() => ($pinned.value = null)}
          >
            ×
          </button>
        </div>
      </div>

      <div class="sessions-inspect-cat">
        <span
          class="sessions-key-dot"
          style={{ background: `rgb(${CATEGORY_RGB[t.category].join(",")})` }}
        />
        {t.category}
        <span class="sessions-inspect-cathelp">{CATEGORY_HELP[t.category]}</span>
      </div>

      <div class="sessions-inspect-scroll">
        {t.promptPreview === undefined ? (
          // record persisted before previews existed — say so, don't guess
          <p class="sessions-inspect-note">
            prompt/response text wasn’t captured for this saved analysis — re-import
            the transcript to dissect it.
          </p>
        ) : (
          t.promptPreview && (
            <div class="sessions-inspect-quote">
              <span class="sessions-inspect-quote-l">serving prompt</span>
              <p>{t.promptPreview}</p>
            </div>
          )
        )}
        {t.textPreview && (
          <div class="sessions-inspect-quote is-response">
            <span class="sessions-inspect-quote-l">response (start)</span>
            <p>{t.textPreview}</p>
          </div>
        )}

        <div class="sessions-stat-row">
          <Stat label="t+" value={fmtSpan(t.tSec)} />
          <Stat label="context" value={fmtTok(t.cacheRead)} />
          <Stat label="new ctx" value={fmtTok(t.cacheWrite)} />
        </div>
        <div class="sessions-stat-row">
          <Stat label="output" value={fmtTok(t.outputTokens)} />
          <Stat label="thinking" value={`${t.thinkingBlocks}`} />
          <Stat label="prose" value={`${t.textLen}ch`} />
        </div>

        {t.tools.length > 0 && (
          <div class="sessions-inspect-tools">
            <span class="sessions-inspect-l">tools, in order</span>
            <div class="sessions-stat-tools">
              {t.tools.map((n, i) => (
                <span key={`${n}-${i}`} class="sessions-tool-chip">
                  {n.split("__").pop()}
                </span>
              ))}
            </div>
          </div>
        )}
        {files.length > 0 && (
          <div class="sessions-inspect-files">
            <span class="sessions-inspect-l">files touched</span>
            <ul>
              {files.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </div>
        )}
        <div class="sessions-inspect-meta">
          {t.model && <span>{shortModel(t.model)}</span>}
          <span title={t.requestId}>{t.requestId.slice(0, 18)}…</span>
        </div>
      </div>
    </div>
  );
}

function SessionsStats(props: { sessions: SessionAnalysis[] }) {
  return (
    <div class="sessions-stats">
      {props.sessions.slice(0, 3).map((a) => (
        <div key={a.id} class="sessions-stat-card">
          <div class="sessions-stat-name">
            {a.name}
            <span class="sessions-stat-fmt">{a.format}</span>
          </div>
          <div class="sessions-stat-row">
            <Stat label="model" value={a.model ?? "—"} />
            <Stat label="responses" value={`${a.nAssistant}`} />
            <Stat label="span" value={fmtSpan(a.spanSec)} />
          </div>
          <div class="sessions-stat-row">
            <Stat
              label={a.outputReliable ? "output" : "output ✧"}
              value={`${fmtTok(a.totalOutput)}`}
            />
            <Stat label="context peak" value={`${fmtTok(a.contextPeak)}`} />
            <Stat label="cache-wr" value={`${fmtTok(a.totalCacheWrite)}`} />
          </div>
          <div class="sessions-stat-row">
            <Stat label="tools" value={`${a.toolTotal}`} />
            <Stat label="files" value={`${a.filesTouched.length}`} />
            {a.subAgentCount > 0 ? (
              <Stat label="sub-agents" value={`${a.subAgentCount} · ${a.sidechainTurns}t`} />
            ) : (
              <Stat label="errors" value={`${a.errorCount}`} />
            )}
          </div>
          {a.toolHistogram.length > 0 && (
            <div class="sessions-stat-tools">
              {a.toolHistogram.slice(0, 5).map(([name, n]) => (
                <span key={name} class="sessions-tool-chip">
                  {name.split("__").pop()} <b>{n}</b>
                </span>
              ))}
            </div>
          )}
          {a.authoritative && <AuthoritativeStrip auth={a.authoritative} />}
          {!a.outputReliable && (
            <p class="sessions-stat-note">
              ✧ per-response output isn’t individually logged in this audit format —
              the total is the authoritative result-line figure; the plotted Z axis uses
              per-turn <b>new-context</b> (cache-write), which is exact.
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

function AuthoritativeStrip(props: { auth: NonNullable<SessionAnalysis["authoritative"]> }) {
  const a = props.auth;
  return (
    <div class="sessions-auth" title="Ground-truth totals from the session's result line">
      <span class="sessions-auth-tag">audit result</span>
      {a.numTurns != null && <span>{a.numTurns} SDK turns</span>}
      {a.costUsd != null && <span>${a.costUsd.toFixed(2)}</span>}
      {a.durationMs != null && <span>{fmtSpan(a.durationMs / 1000)}</span>}
      {a.models.length > 0 && (
        <span class="sessions-auth-models">{a.models.map((m) => shortModel(m)).join(" + ")}</span>
      )}
    </div>
  );
}

function fmtSpan(sec: number): string {
  if (sec >= 3600) return `${(sec / 3600).toFixed(1)}h`;
  if (sec >= 90) return `${(sec / 60).toFixed(1)}m`;
  return `${Math.round(sec)}s`;
}

function shortModel(m: string): string {
  return m
    .replace(/^claude-/, "")
    .replace(/-\d{8}$/, "")
    .replace(/-(\d)$/, ".$1");
}

function Stat(props: { label: string; value: string }) {
  return (
    <div class="sessions-stat">
      <span class="sessions-stat-v">{props.value}</span>
      <span class="sessions-stat-l">{props.label}</span>
    </div>
  );
}

// ── helpers ────────────────────────────────────────────────────────────────

function doParse(raw: string, name: string, errorMsg: { value: string }) {
  try {
    const a = parseSessionTranscript(raw, cleanName(name));
    if (a.turns.length === 0) {
      errorMsg.value = "no model responses found — is this a Claude Code .jsonl transcript?";
      return;
    }
    appStore.getState().addSessionAnalysis(a);
    saveSessionAnalysis(a).catch(() => {});
    errorMsg.value = "";
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : String(e);
  }
}

function loadSample() {
  const a = parseSessionTranscript(SAMPLE_TRANSCRIPT, "sample session");
  appStore.getState().addSessionAnalysis(a);
  saveSessionAnalysis(a).catch(() => {});
}

function cleanName(name: string): string {
  return name.replace(/\.(jsonl|json|txt)$/i, "").slice(0, 60);
}

function swatchStyle(i: number): Record<string, string> {
  const h = (i * 137.508) % 360;
  return { background: `hsl(${h.toFixed(0)} 66% 62%)` };
}

function fmtTok(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return `${Math.round(n)}`;
}
