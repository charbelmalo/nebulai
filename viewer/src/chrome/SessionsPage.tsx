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
  buildAgentGraph,
  CATEGORY_ORDER,
  parseSessionTranscript,
  type SessionAnalysis,
  type SessionTurn,
  type ToolCategory,
  type ToolOutcome,
} from "./sessionlog";
import { Stat } from "./StatStrip";
import { SessionFieldDriver, type PlaybackState } from "../scene/sessions/SessionFieldDriver";
import { categoryColor } from "../scene/sessions/appearance";
import { asinhScale, suggestK } from "../scene/sessions/scales";
import {
  deleteSessionAnalysis,
  loadAllSessionAnalyses,
  saveSessionAnalysis,
} from "./sessionStore";
import { $appearance, $sessions, $settings } from "./state";

/** Pinned turn = the node last clicked in the plot. Page-level (not persisted):
 *  the inspector, the plot highlight, and keyboard nav all read/write it. */
const $pinned = signal<{ sessionId: string; index: number } | null>(null);
/** Categories currently dimmed in the plot (legend-key toggles). */
const $dimmedCats = signal<ToolCategory[]>([]);
/** Which axes the driver actually eased (asinh) for the current data. Only the
 *  driver knows — a mild spread stays linear — and the legend must not claim a
 *  compression that didn't happen. */
const $easedAxes = signal<string[]>([]);
/** The live driver, published so the transport (a sibling of the plot) can drive
 *  it. Null until the GPU init resolves / after teardown. */
const $driver = signal<SessionFieldDriver | null>(null);
/** Latest timeline-transport snapshot, emitted by the driver as the sweep
 *  advances. Drives the play/pause button, scrubber, and time readout. */
const $playback = signal<PlaybackState | null>(null);

const CATEGORY_HELP: Record<ToolCategory, string> = {
  orient: "read / search / fetch",
  plan: "task lifecycle, plan mode",
  edit: "write / edit files",
  exec: "run commands",
  deliver: "present / publish / notify",
  reflect: "pure text or thinking",
};

/** A hand-written stand-in for a real `.jsonl`, so the page has something to
 *  show before you drop a transcript on it.
 *
 *  Every `tool_use` carries an `id` and every one but the last is answered by a
 *  `tool_result` quoting it, because that is what a real transcript looks like
 *  and the outcome split reads exactly those links. An earlier version of this
 *  fixture omitted both, which was invisible until the split existed and then
 *  rendered a demo session whose every call was unresolved. The failing
 *  `npm test` and the trailing unanswered call are deliberate: they are the
 *  only two states a clean sample would never otherwise exercise. */
const SAMPLE_TRANSCRIPT = [
  `{"type":"user","timestamp":"2026-07-09T18:00:00Z","message":{"role":"user","content":"Refactor the auth module and add tests."}}`,
  `{"type":"assistant","requestId":"req-a","timestamp":"2026-07-09T18:00:12Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":40,"output_tokens":180,"cache_read_input_tokens":12000,"cache_creation_input_tokens":8000},"content":[{"type":"thinking","thinking":"look at the module"}]}}`,
  `{"type":"assistant","requestId":"req-a","timestamp":"2026-07-09T18:00:12Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":40,"output_tokens":180,"cache_read_input_tokens":12000,"cache_creation_input_tokens":8000},"content":[{"type":"tool_use","id":"tu-1","name":"Read","input":{"file_path":"src/auth.ts"}}]}}`,
  `{"type":"user","timestamp":"2026-07-09T18:00:20Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu-1","content":"...file contents..."}]}}`,
  `{"type":"assistant","requestId":"req-b","timestamp":"2026-07-09T18:00:40Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":30,"output_tokens":900,"cache_read_input_tokens":22000,"cache_creation_input_tokens":3000},"content":[{"type":"tool_use","id":"tu-2","name":"Edit","input":{"file_path":"src/auth.ts"}}]}}`,
  `{"type":"user","timestamp":"2026-07-09T18:00:52Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu-2","content":"edited"}]}}`,
  `{"type":"assistant","requestId":"req-c","timestamp":"2026-07-09T18:01:20Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":20,"output_tokens":520,"cache_read_input_tokens":30000,"cache_creation_input_tokens":1500},"content":[{"type":"tool_use","id":"tu-3","name":"Bash","input":{"command":"npm test"}}]}}`,
  `{"type":"user","timestamp":"2026-07-09T18:01:34Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu-3","is_error":true,"content":"1 failing: expired token still accepted"}]}}`,
  `{"type":"assistant","requestId":"req-d","timestamp":"2026-07-09T18:01:50Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":10,"output_tokens":260,"cache_read_input_tokens":33000,"cache_creation_input_tokens":600},"content":[{"type":"tool_use","id":"tu-4","name":"Edit","input":{"file_path":"src/auth.ts"}}]}}`,
  `{"type":"user","timestamp":"2026-07-09T18:02:01Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu-4","content":"edited"}]}}`,
  `{"type":"assistant","requestId":"req-e","timestamp":"2026-07-09T18:02:18Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":12,"output_tokens":300,"cache_read_input_tokens":35000,"cache_creation_input_tokens":700},"content":[{"type":"tool_use","id":"tu-5","name":"Bash","input":{"command":"npm test"}}]}}`,
  `{"type":"user","timestamp":"2026-07-09T18:02:31Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu-5","content":"12 passing"}]}}`,
  `{"type":"assistant","requestId":"req-f","timestamp":"2026-07-09T18:02:44Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":14,"output_tokens":190,"cache_read_input_tokens":36000,"cache_creation_input_tokens":400},"content":[{"type":"tool_use","id":"tu-6","name":"Read","input":{"file_path":"src/auth.test.ts"}}]}}`,
  `{"type":"assistant","requestId":"req-g","timestamp":"2026-07-09T18:02:58Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":10,"output_tokens":260,"cache_read_input_tokens":37000,"cache_creation_input_tokens":300},"content":[{"type":"text","text":"Done — auth refactored and tests pass."}]}}`,
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
          {active.length > 0 && <SessionTransport />}
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
  const driverRef = useRef<SessionFieldDriver | null>(null);
  const ready = useSignal(false);

  // init once
  useEffect(() => {
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    const host = canvas?.parentElement;
    if (!canvas || !overlay || !host) return;
    let disposed = false;
    const driver = new SessionFieldDriver();
    driver.onSelect = (sel) => {
      $pinned.value = sel;
    };
    driver.onPlayback = (s) => {
      $playback.value = s;
    };
    driver
      .init(canvas, overlay)
      .then(() => {
        if (disposed) {
          driver.dispose();
          return;
        }
        driverRef.current = driver;
        $driver.value = driver;
        window.__sessionDriver = driver; // e2e + debugging handle, as with __driver
        driver.setReducedMotion(appStore.getState().settings.reducedMotion);
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
      $driver.value = null;
      $playback.value = null;
      delete window.__sessionDriver;
    };
  }, []);

  // push data whenever the active set changes
  const key = props.analyses.map((a) => a.id).join("|");
  useEffect(() => {
    if (!ready.value) return;
    driverRef.current?.setSessions(props.analyses);
    const d = driverRef.current?.describe();
    $easedAxes.value = d
      ? (["x", "y", "z"] as const)
          .filter((ax) => d.curved[ax])
          .map((ax) => ({ x: "time", y: "context", z: "new-context" })[ax])
      : [];
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

  // live appearance — Settings › Appearance › Sessions. Most knobs are uniforms
  // (no rebuild); an axis-mode change reshapes the field, so recompute which
  // axes ended up eased for the legend's honesty note.
  const appearance = $appearance.value.sessions;
  useEffect(() => {
    const d = driverRef.current;
    if (!ready.value || !d) return;
    d.setAppearance(appearance);
    const desc = d.describe();
    $easedAxes.value = (["x", "y", "z"] as const)
      .filter((ax) => desc.curved[ax])
      .map((ax) => ({ x: "time", y: "context", z: "new-context" })[ax]);
  }, [appearance, ready.value]);

  // global Settings › bloom toggle (webgpu rung only)
  const bloom = $settings.value.bloom;
  useEffect(() => {
    if (ready.value) driverRef.current?.setBloom(bloom);
  }, [bloom, ready.value]);

  // global Settings › reduced motion — kills the wavefront flare (the reveal
  // itself still plays, just without the pulse)
  const reducedMotion = $settings.value.reducedMotion;
  useEffect(() => {
    if (ready.value) driverRef.current?.setReducedMotion(reducedMotion);
  }, [reducedMotion, ready.value]);

  return (
    // req 10 — keyboard focus + ARIA readout for the 3-D GPU plot. The host is
    // the tab stop (no per-node DOM); exact turn values are announced by the
    // driver's aria-live tooltip on hover. Focus ring lives in CSS.
    <div
      class="sessions-canvas-host"
      tabIndex={0}
      role="img"
      onKeyDown={(e) => {
        // Space toggles the timeline (video-player convention); don't scroll
        if ((e.key === " " || e.key === "Spacebar") && props.analyses.length > 0) {
          e.preventDefault();
          driverRef.current?.togglePlay();
        }
      }}
      aria-label={
        props.analyses.length
          ? `3-D session trajectory plot: ${props.analyses.length} session${
              props.analyses.length === 1 ? "" : "s"
            }. Each turn is placed by wall-clock time, context size, and new context this turn. Hover a node for exact values. Press space to play the session as a timeline.`
          : "3-D session trajectory plot (no sessions loaded)"
      }
    >
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
  const eased = $easedAxes.value;
  const cfg = $appearance.value.sessions;
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
                  style={{ background: categoryColor(cfg, c) }}
                />
                <span class="sessions-key-label">{c}</span>
                <span class="sessions-key-help">{CATEGORY_HELP[c]}</span>
              </button>
            </li>
          );
        })}
      </ul>
      <div class="sessions-legend-axes">
        X time · Y context (cache-read) · Z new-context/turn (cache-write) · glow = output
        tokens (ranked) · size = tools used · red = a tool failed · faded = sub-agent ·
        trail = turn order, lights up around what you point at · drag to orbit ·{" "}
        <b>click a node to dissect it</b>
        {eased.length > 0 && (
          <>
            <br />
            <span class="sessions-legend-scale">
              {eased.join(", ")} {eased.length > 1 ? "axes are" : "axis is"} eased so the
              quiet turns aren't crushed against the wall — spacing changes, readings
              don't. Tick labels and tooltip values are exact.
            </span>
          </>
        )}
      </div>
    </div>
  );
}

// ── timeline transport — play the session start→finish ─────────────────────

function SessionTransport() {
  const d = $driver.value;
  const pb = $playback.value;
  if (!d || !pb || !pb.hasData) return null;
  return (
    <div class="transport sessions-transport" role="group" aria-label="Session timeline playback">
      <button
        type="button"
        class="tp-btn tp-play"
        // at rest playhead is parked at the end, and play() restarts from the top,
        // so a plain play glyph reads truer than a replay one
        aria-label={pb.playing ? "Pause timeline" : "Play timeline from start"}
        onClick={() => d.togglePlay()}
      >
        {pb.playing ? "⏸" : "▶"}
      </button>
      <button
        type="button"
        class="tp-btn"
        aria-label="Restart from beginning"
        title="Restart"
        onClick={() => d.restart()}
      >
        ↺
      </button>
      <input
        class="tp-scrub"
        type="range"
        min={0}
        max={1}
        step={0.001}
        value={pb.playhead}
        aria-label="Timeline position"
        aria-valuetext={`${fmtSpan(pb.tSec)} of ${fmtSpan(pb.totalSec)}, ${pb.revealed} of ${pb.total} turns`}
        // dragging the scrubber pauses, then seeks — so it never fights the sweep
        onInput={(e) => {
          const v = Number((e.currentTarget as HTMLInputElement).value);
          d.pause();
          d.seek(v);
        }}
        style={{
          "--tp-progress": `${(pb.playhead * 100).toFixed(1)}%`,
        }}
      />
      <span class="tp-time" aria-hidden="true">
        <span class="tp-now">{fmtSpan(pb.tSec)}</span>
        <span class="tp-sep">/</span>
        <span class="tp-total">{fmtSpan(pb.totalSec)}</span>
      </span>
      <span class="tp-count" title="turns revealed by the wavefront">
        {pb.revealed}
        <i>/{pb.total}</i>
      </span>
      <label class="tp-speed">
        <span class="visually-hidden">Playback speed</span>
        <select
          value={String(pb.speed)}
          onChange={(e) => d.setSpeed(Number((e.currentTarget as HTMLSelectElement).value))}
        >
          <option value="0.5">0.5×</option>
          <option value="1">1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
        </select>
      </label>
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
  const cfg = $appearance.value.sessions;
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
          style={{ background: categoryColor(cfg, t.category) }}
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

        <div class="stat-row">
          <Stat label="t+" value={fmtSpan(t.tSec)} />
          <Stat label="context" value={fmtTok(t.cacheRead)} />
          <Stat label="new ctx" value={fmtTok(t.cacheWrite)} />
        </div>
        <div class="stat-row">
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
          <div class="stat-row">
            <Stat label="model" value={a.model ?? "—"} />
            <Stat label="responses" value={`${a.nAssistant}`} />
            <Stat label="span" value={fmtSpan(a.spanSec)} />
          </div>
          <div class="stat-row">
            <Stat
              label={a.outputReliable ? "output" : "output ✧"}
              value={`${fmtTok(a.totalOutput)}`}
            />
            <Stat label="context peak" value={`${fmtTok(a.contextPeak)}`} />
            <Stat label="cache-wr" value={`${fmtTok(a.totalCacheWrite)}`} />
          </div>
          <div class="stat-row">
            <Stat label="tools" value={`${a.toolTotal}`} />
            <Stat label="files" value={`${a.filesTouched.length}`} />
            {a.subAgentCount > 0 ? (
              <Stat label="sub-agents" value={`${a.subAgentCount} · ${a.sidechainTurns}t`} />
            ) : (
              <Stat label="errors" value={`${a.errorCount}`} />
            )}
          </div>
          {a.toolOutcomes.length > 0 ? (
            <ToolOutcomeChart outcomes={a.toolOutcomes} unattributed={a.unattributedErrors} />
          ) : (
            a.toolTotal > 0 && (
              <p class="sessions-stat-note sessions-stat-note-top">
                This analysis was saved before per-tool outcomes were recorded, and the raw
                transcript isn’t kept — load the <code>.jsonl</code> again to get the split.
              </p>
            )
          )}
          <PaceChart turns={a.turns} />
          {a.context ? (
            <ContextChart ctx={a.context} />
          ) : (
            <p class="sessions-stat-note sessions-stat-note-top">
              This analysis predates the context breakdown and its turns weren’t stored either —
              load the <code>.jsonl</code> again to decompose the window.
            </p>
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

/** How many tools get a bar. Everything past this is summed into a one-line
 *  remainder — stated, never dropped: a card that shows six bars and stops is
 *  indistinguishable from a session that only used six tools. */
const OUTCOME_ROWS = 6;

/** Per-tool outcome bars, replacing the old count-only chip row.
 *
 *  Each bar is one tool's calls, stacked ok → failed → unresolved, on a track
 *  scaled to the busiest tool in the session. Shared scale, not per-row
 *  normalisation: the point of stacking them is to compare, and a bar that
 *  fills its own row regardless of count compares nothing.
 *
 *  `unresolved` earns its own colour rather than being blank track. It means a
 *  `tool_use` that never got a `tool_result` — interrupted, or the log ends
 *  mid-call — which is a real thing that happened, not missing data. Blanking
 *  it would quietly shorten the bar and inflate the apparent success rate. */
function ToolOutcomeChart(props: { outcomes: ToolOutcome[]; unattributed: number }) {
  const all = props.outcomes;
  const rows = all.slice(0, OUTCOME_ROWS);
  const tail = all.slice(OUTCOME_ROWS);
  // sorted descending by the fold, so the head is the max
  const max = Math.max(rows[0]?.total ?? 0, 1);
  const sum = (k: "ok" | "failed" | "unresolved") => all.reduce((s, o) => s + o[k], 0);
  const totals = { ok: sum("ok"), failed: sum("failed"), unresolved: sum("unresolved") };
  // The remainder gets a sentence, not a bar. A pool of seventeen tools is not
  // a peer of `Bash` — drawn on the same axis it routinely outgrows every named
  // tool (measured: 213 calls against the busiest tool's 40) and squashes the
  // rows the chart exists to show, while inviting a comparison that is a
  // category error anyway.
  const rest = tail.length
    ? {
        tools: tail.length,
        ok: tail.reduce((s, o) => s + o.ok, 0),
        failed: tail.reduce((s, o) => s + o.failed, 0),
        unresolved: tail.reduce((s, o) => s + o.unresolved, 0),
        total: tail.reduce((s, o) => s + o.total, 0),
      }
    : null;

  return (
    <div class="tool-outcomes">
      <div class="tool-outcome-legend">
        <span class="tool-outcome-key is-ok" title="tool_result returned without is_error">
          ok <b>{totals.ok}</b>
        </span>
        <span class="tool-outcome-key is-failed" title="tool_result flagged is_error">
          failed <b>{totals.failed}</b>
        </span>
        <span
          class="tool-outcome-key is-unresolved"
          title="the call was made but no tool_result ever came back — interrupted, or the log ends mid-call"
        >
          unresolved <b>{totals.unresolved}</b>
        </span>
      </div>
      <ul class="tool-outcome-rows">
        {rows.map((r) => (
          <li class="tool-outcome-row" key={r.tool}>
            <span class="tool-outcome-name" title={r.tool}>
              {r.tool.split("__").pop()}
            </span>
            <span
              class="tool-outcome-bar"
              title={`${r.tool}: ${r.ok} ok · ${r.failed} failed · ${r.unresolved} unresolved`}
            >
              {(["ok", "failed", "unresolved"] as const).map((k) =>
                r[k] > 0 ? (
                  <span
                    key={k}
                    class={`tool-outcome-seg is-${k}`}
                    style={{ width: `${(r[k] / max) * 100}%` }}
                  />
                ) : null,
              )}
            </span>
            <span class="tool-outcome-n">{r.total}</span>
          </li>
        ))}
      </ul>
      {rest && (
        <p class="tool-outcome-rest">
          <b>{rest.tools}</b> more {rest.tools === 1 ? "tool" : "tools"}, not drawn:{" "}
          <b>{rest.total}</b> calls — {rest.ok} ok, {rest.failed} failed, {rest.unresolved}{" "}
          unresolved
        </p>
      )}
      {props.unattributed > 0 && (
        <p class="sessions-stat-note">
          <b>{props.unattributed}</b> further{" "}
          {props.unattributed === 1 ? "failure is" : "failures are"} counted in this session but
          absent from the bars above: the audit format reports them without a{" "}
          <b>tool_use_id</b>, so there is no tool to charge them to.
        </p>
      )}
    </div>
  );
}

/** Context utilization (#4).
 *
 *  One column per main-agent turn, stacked bottom-up **reused → new → fresh**,
 *  on a track scaled to the session's largest prompt. Height is that turn's
 *  whole context window; the split is where the tokens came from as far as the
 *  cache is concerned, which is the part a transcript actually meters.
 *
 *  WHAT THIS IS NOT. The feature asked to segment the window into
 *  system / user / output / retrieved. Three of those four cannot be separated
 *  from a Claude Code transcript — it meters tokens per request, not per
 *  content block — and the fourth does not exist in this repo at all (no RAG).
 *
 *  The line below the columns therefore reports growth and model output as TWO
 *  measured totals, and explicitly declines to subtract one from the other. The
 *  subtraction is the trap: it reads as "what the tools contributed" and it is
 *  not, because tool output enters the window without being billed as output
 *  while thinking is billed as output and never re-fed. Measured on a real
 *  731-turn transcript, that difference sums to −351k tokens.
 *
 *  A shared scale, not per-column normalisation: the whole point is to see the
 *  window grow, and a column that fills its own track regardless of size shows
 *  no growth at all. */
/** Where a session's time actually went — the "per-step timing" half of
 *  requested feature #2.
 *
 *  The field already draws the path and the tooltip already names each step's
 *  gap, but a 700-node cloud does not answer "which steps were the slow ones"
 *  by hover. This strip does: one cell per measured gap, in step order.
 *
 *  Two scoping decisions worth stating rather than burying. The cells are
 *  **asinh**-scaled, the same treatment the field's axes use and for the same
 *  reason — step gaps run from 0.4s to twenty minutes, and a linear strip would
 *  be one spike and several hundred invisible cells. And the concentration
 *  figure is over **measured step time**, not wall clock: when more than one
 *  agent ran, their gaps overlap in real time and summing them would over-count
 *  the session. On every real transcript here there is exactly one agent and
 *  the two coincide, but the label has to hold in the case that doesn't. */
const PACE_MEDIAN_AT = 0.2;

function PaceChart(props: { turns: SessionTurn[] }) {
  const g = buildAgentGraph(props.turns);
  if (!g.timed) {
    if (props.turns.length === 0) return null;
    // Two different reasons produce no gaps, and conflating them would report a
    // data problem where there is none. 204 of the 239 real transcripts here
    // have no gap for the ordinary reason: they are one turn long.
    const single = g.paths.every((p) => p.steps.length < 2);
    return (
      <p class="sessions-stat-note sessions-stat-note-top">
        {single
          ? "One step per agent — a gap needs two responses to measure."
          : "No two consecutive responses both carried a timestamp, so no step could be timed."}
      </p>
    );
  }
  const steps = g.paths
    .flatMap((p) => p.steps)
    .filter((s) => s.gapSec !== null)
    .sort((x, y) => x.turn - y.turn);
  const gaps = steps.map((s) => s.gapSec!);
  const total = gaps.reduce((a, b) => a + b, 0);
  const max = gaps.reduce((a, b) => (b > a ? b : a), 0);
  // Median at 0.2 of the strip, not the 0.45 a 3-D axis wants: a bar chart of a
  // heavy tail should sit low and let the slow steps stand up. Measured — see
  // `suggestK`. `force` because a strip is worth bending even on a mild spread.
  const scale = asinhScale(max, suggestK(gaps, max, true, PACE_MEDIAN_AT));

  // How few steps account for half the measured time. The heavy tail is the
  // finding: on a long session a handful of steps own most of the clock.
  const sorted = [...gaps].sort((a, b) => b - a);
  let half = 0;
  let acc = 0;
  while (half < sorted.length && acc < total / 2) acc += sorted[half++]!;

  const slowest = steps.reduce((a, b) => (b.gapSec! > a.gapSec! ? b : a));
  const multi = g.paths.length > 1;

  return (
    <div class="pace-util">
      <div class="ctx-legend">
        <span class="ctx-scope">between steps</span>
        <span class="ctx-key is-pace" title="the largest gap between two consecutive responses">
          longest <b>{fmtSpan(max)}</b>
        </span>
        <span class="ctx-key is-pace-sum" title="every measured gap, summed">
          measured <b>{fmtSpan(total)}</b>
        </span>
      </div>
      <div
        class="pace-cells"
        role="img"
        aria-label={`gap before each of ${steps.length} steps, asinh-scaled`}
      >
        {steps.map((s) => (
          <span
            key={`${s.agentId}:${s.turn}`}
            class={`pace-cell${s.turn === slowest.turn ? " is-slowest" : ""}`}
            style={{ height: `${Math.max(scale.toUnit(s.gapSec!) * 100, 2)}%` }}
            title={
              `turn ${s.turn + 1} — ${fmtSpan(s.gapSec!)} since the previous step\n` +
              (multi ? `${s.agentId === "main" ? "main" : "sub-agent"} step ${s.step} of ${s.ofSteps}` : `step ${s.step} of ${s.ofSteps}`)
            }
          />
        ))}
      </div>
      <p class="ctx-growth">
        Half the measured step time falls in <b>{half}</b> of {steps.length}{" "}
        {steps.length === 1 ? "gap" : "gaps"}; the longest is <b>{fmtSpan(max)}</b> before turn{" "}
        <b>{slowest.turn + 1}</b>. A gap is elapsed time between two responses — tool work, model
        latency and, after a human prompt, the human — not how long the step itself took.
      </p>
      {multi && (
        <p class="sessions-stat-note">
          <b>{g.paths.length}</b> agents ran here, so each step is numbered and timed within its own
          agent and the totals above are summed step time, not wall clock — concurrent agents
          overlap.
        </p>
      )}
    </div>
  );
}

function ContextChart(props: { ctx: NonNullable<SessionAnalysis["context"]> }) {
  const c = props.ctx;
  const n = c.slices.length;
  if (n === 0) {
    return (
      <p class="sessions-stat-note sessions-stat-note-top">
        No main-agent turns carried usage, so there is no window to decompose.
        {c.excludedSidechain > 0 &&
          ` ${c.excludedSidechain} sub-agent ${c.excludedSidechain === 1 ? "turn ran" : "turns ran"} in a separate context.`}
      </p>
    );
  }
  const peak = Math.max(c.peakPrompt, 1);
  const last = c.slices[n - 1]!;
  const compacted = new Set(c.compactedAt);
  // Summed over the steps that GREW, with compactions left out: a compaction
  // drops ~150k in one step and would otherwise swamp every real number here.
  // It gets its own sentence instead.
  let grew = 0;
  let wrote = 0;
  let steps = 0;
  for (const s of c.slices) {
    if (s.growth === null || compacted.has(s.turn)) continue;
    grew += s.growth;
    wrote += s.priorOutput ?? 0;
    steps++;
  }
  // How often the window grew by LESS than the model had just written. Not an
  // error: thinking is billed as output and then dropped from the next prompt.
  const shortfall = c.slices.filter(
    (s) => !compacted.has(s.turn) && s.residual !== null && s.residual < 0,
  ).length;
  const KEYS = [
    { k: "reused", label: "reused", hint: "already in the cache — the conversation so far" },
    { k: "written", label: "new", hint: "written into the cache for the first time this turn" },
    { k: "fresh", label: "fresh", hint: "sent uncached" },
  ] as const;
  // The legend reads the LAST turn, not the session — three numbers under a
  // bare "reused / new / fresh" invite being read as totals, which they aren't.
  //
  // The two halves have to count the same thing. `last.turn + 1` is a position
  // among ALL turns while `n` counts only the drawn (main-agent) ones, which
  // read "turn 7 of 4" the first time a session with a sub-agent was rendered.
  // The numerator is worth keeping — a session turn number is what joins this
  // label to the field and to the column tooltips — so the denominator moves to
  // match it. `n + excludedSidechain` is the session's turn count, and on the
  // single-agent sessions that make up every real transcript here the second
  // term is 0 and the label is what it always was.
  const scope = `turn ${last.turn + 1} of ${n + c.excludedSidechain}`;

  return (
    <div class="ctx-util">
      <div class="ctx-legend">
        <span class="ctx-scope">at {scope}</span>
        {KEYS.map((d) => (
          <span key={d.k} class={`ctx-key is-${d.k}`} title={`${d.hint} — at ${scope}`}>
            {d.label} <b>{fmtTok(last[d.k])}</b>
          </span>
        ))}
        <span class="ctx-key is-peak" title="largest prompt any single turn was sent">
          peak <b>{fmtTok(c.peakPrompt)}</b>
        </span>
      </div>
      <div class="ctx-cols" role="img" aria-label={`context window across ${n} turns`}>
        {c.slices.map((s) => (
          <span
            key={s.turn}
            class={`ctx-col${compacted.has(s.turn) ? " is-compacted" : ""}`}
            style={{ height: `${(s.prompt / peak) * 100}%` }}
            title={
              `turn ${s.turn + 1} — ${fmtTok(s.prompt)} in context\n` +
              `${fmtTok(s.reused)} reused · ${fmtTok(s.written)} new · ${fmtTok(s.fresh)} fresh\n` +
              (s.growth === null
                ? "first turn — nothing to difference against"
                : `${s.growth >= 0 ? "+" : "−"}${fmtTok(Math.abs(s.growth))} vs the turn before` +
                  (s.priorOutput !== null
                    ? `; the model had just written ${fmtTok(s.priorOutput)}`
                    : ""))
            }
          >
            {KEYS.map((d) =>
              s[d.k] > 0 ? (
                <span
                  key={d.k}
                  class={`ctx-seg is-${d.k}`}
                  style={{ height: `${(s[d.k] / s.prompt) * 100}%` }}
                />
              ) : null,
            )}
          </span>
        ))}
      </div>
      {steps > 0 && (
        <p class="ctx-growth">
          Over {steps} growing {steps === 1 ? "step" : "steps"} the window gained{" "}
          <b>{fmtTok(grew)}</b> while the model wrote <b>{fmtTok(wrote)}</b>. Two measurements, not
          a split: what a tool returns enters the window without ever being billed as output, and
          thinking is billed as output but never fed back
          {shortfall > 0 && (
            <>
              {" "}
              — which is why on <b>{shortfall}</b> of {steps === 1 ? "that step" : "those steps"} the
              window grew by less than the model had just written
            </>
          )}
          .
        </p>
      )}
      {c.compactedAt.length > 0 && (
        <p class="sessions-stat-note">
          The window <b>shrank</b> at {c.compactedAt.length === 1 ? "turn" : "turns"}{" "}
          {c.compactedAt.map((t) => t + 1).join(", ")} — compaction, or a context reset. Held out of
          the totals above, where a single 150k drop would swamp every other number, but kept in the
          columns, which show the window as it really was.
        </p>
      )}
      {c.excludedSidechain > 0 && (
        <p class="sessions-stat-note">
          <b>{c.excludedSidechain}</b> sub-agent{" "}
          {c.excludedSidechain === 1 ? "turn is" : "turns are"} not drawn: each sub-agent runs in
          its own window, and differencing across it would measure the gap between two unrelated
          conversations.
        </p>
      )}
      {c.exact === false && (
        <p class="sessions-stat-note">
          At least one response reported different prompt usage on different lines, so each column’s
          three parts sum to an <b>upper bound</b> on that turn’s prompt rather than to a figure any
          single request was sent.
        </p>
      )}
      {c.exact === null && (
        <p class="sessions-stat-note">
          Rebuilt from a stored analysis. The per-turn token counts are the real ones; the check for
          whether a response reported its prompt consistently wasn’t saved, so it is unknown here
          rather than assumed either way.
        </p>
      )}
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

/* `Stat` now lives in ./StatStrip — the interp charts needed the same tile. */

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
