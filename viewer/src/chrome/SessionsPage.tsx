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

import { useSignal } from "@preact/signals";
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
  }, [key, ready.value]);

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
  return (
    <div class="sessions-legend">
      <div class="sessions-legend-title">turn category</div>
      <ul class="sessions-legend-keys">
        {CATEGORY_ORDER.map((c) => (
          <li key={c}>
            <span class="sessions-key-dot" style={{ background: `rgb(${CATEGORY_RGB[c].join(",")})` }} />
            <span class="sessions-key-label">{c}</span>
            <span class="sessions-key-help">{CATEGORY_HELP[c]}</span>
          </li>
        ))}
      </ul>
      <div class="sessions-legend-axes">
        X time · Y context (cache-read) · Z new-context/turn (cache-write) · size ∝ tools ·
        faded = sub-agent · drag to orbit
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
