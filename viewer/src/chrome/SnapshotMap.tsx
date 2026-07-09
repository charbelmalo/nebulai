/** Snapshot Map — a per-task, per-topic conversation-log visualiser. Drop
 *  a JSON agent log, pick a topic (design / shaders / interaction / …), and
 *  scrub the turn slider to see which keywords co-occurred and how the
 *  connections thickened as the conversation unfolded. */

import { useSignal } from "@preact/signals";
import { useEffect, useMemo, useRef } from "preact/hooks";
import { appStore } from "../app/store";
import { $snapshot } from "./state";
import { analyzeSnapshot, layoutRadial, parseConversationText } from "./snapshot";

const SAMPLE_LOG_TEXT = JSON.stringify(
  {
    messages: [
      {
        role: "user",
        content:
          "Let's rework the atlas hover states. The hit target feels tight and the focus outline is inconsistent.",
      },
      {
        role: "assistant",
        content:
          "Good call — we'll bump the hit target to 44px, unify focus, and align spacing. I'll also revisit the empty state so the layout hierarchy reads.",
      },
      {
        role: "user",
        content:
          "While you're in there — the bloom pass is way too hot. Can we tone the fresnel and add a subtle vignette?",
      },
      {
        role: "assistant",
        content:
          "Yes. TSL bloom threshold going up, fresnel curve softened, vignette baked into the post-processing chain. Compute uniforms should stay untouched.",
      },
      {
        role: "user",
        content:
          "One more — the submenu keeps closing on my diagonal. Add safe triangle and make sure escape closes it before the parent.",
      },
      {
        role: "assistant",
        content:
          "Adding safe triangle + aria-activedescendant for the combobox pattern. Escape now unwinds innermost-first. Arrow keys clamped, no roving tabindex.",
      },
    ],
  },
  null,
  2,
);

export function SnapshotMap() {
  const snap = $snapshot.value;
  const activeLog = snap.logs.find((l) => l.id === snap.activeLogId) ?? null;
  const activeTopic = snap.topics.find((t) => t.id === snap.activeTopicId) ?? null;
  const dragOver = useSignal(false);

  const analysis = useMemo(
    () => analyzeSnapshot(activeLog, activeTopic, snap.turnIndex),
    [activeLog, activeTopic, snap.turnIndex],
  );

  const currentTurn = activeLog?.turns[snap.turnIndex] ?? null;
  const maxTurn = activeLog ? activeLog.turns.length - 1 : 0;

  useEffect(() => {
    if (!snap.playing || !activeLog) return;
    const t = window.setInterval(() => {
      const s = appStore.getState();
      const log = s.snapshot.logs.find((l) => l.id === s.snapshot.activeLogId);
      if (!log) return;
      if (s.snapshot.turnIndex >= log.turns.length - 1) {
        s.setPlaying(false);
      } else {
        s.setTurnIndex(s.snapshot.turnIndex + 1);
      }
    }, 900);
    return () => clearInterval(t);
  }, [snap.playing, activeLog?.id]);

  return (
    <div class="snapshot-page" role="main">
      <div class="snapshot-shell">
        <SnapshotLeft dragOver={dragOver.value} setDragOver={(v) => (dragOver.value = v)} />
        <div class="snapshot-stage">
          <SnapshotGraph
            keywords={activeTopic?.keywords ?? []}
            active={new Set(analysis.activeKeywords)}
            edges={analysis.edges}
            totals={analysis.totals}
          />
          {activeLog && (
            <div class="snapshot-turn-card">
              <div class="snapshot-turn-head">
                <span class="snapshot-turn-role" data-role={currentTurn?.role}>
                  {currentTurn?.role ?? "—"}
                </span>
                <span class="snapshot-turn-pos">
                  turn {snap.turnIndex + 1} / {activeLog.turns.length}
                </span>
              </div>
              <p class="snapshot-turn-text">
                {truncate(currentTurn?.text ?? "", 240)}
              </p>
              <div class="snapshot-turn-chips">
                {[...(analysis.perTurn.get(snap.turnIndex) ?? [])].map((k) => (
                  <span key={k} class="snapshot-chip is-hot">
                    {k}
                  </span>
                ))}
              </div>
            </div>
          )}
          {!activeLog && (
            <div class="snapshot-empty">
              <h2>Drop a conversation log</h2>
              <p>
                Paste JSON on the left, drop a file anywhere, or load the sample to see how the
                connections evolve turn by turn.
              </p>
              <button
                type="button"
                class="btn-primary"
                onClick={() => loadSample()}
              >
                Load sample conversation
              </button>
            </div>
          )}
        </div>
      </div>

      <footer class="snapshot-footer">
        <SnapshotScrubber
          disabled={!activeLog}
          value={snap.turnIndex}
          max={maxTurn}
          onChange={(i) => appStore.getState().setTurnIndex(i)}
        />
      </footer>
    </div>
  );
}

function SnapshotLeft(props: { dragOver: boolean; setDragOver: (v: boolean) => void }) {
  const snap = $snapshot.value;
  const activeTopic = snap.topics.find((t) => t.id === snap.activeTopicId) ?? null;
  const pasteText = useSignal("");
  const pasteName = useSignal("");
  const fileRef = useRef<HTMLInputElement>(null);
  const errorMsg = useSignal("");

  const doParse = (raw: string, name: string) => {
    try {
      const log = parseConversationText(raw, name || `log-${snap.logs.length + 1}`);
      appStore.getState().addSnapshotLog(log);
      errorMsg.value = "";
      pasteText.value = "";
      pasteName.value = "";
    } catch (e) {
      errorMsg.value = e instanceof Error ? e.message : String(e);
    }
  };

  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    props.setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      const text = await file.text();
      doParse(text, file.name);
    }
  };

  return (
    <aside class="snapshot-side">
      <section class="snapshot-side-block">
        <h3>Topic</h3>
        <div class="snapshot-topic-list">
          {snap.topics.map((t) => (
            <button
              key={t.id}
              type="button"
              class={`snapshot-topic${t.id === snap.activeTopicId ? " is-active" : ""}`}
              onClick={() => appStore.getState().setActiveTopic(t.id)}
            >
              <span class="snapshot-topic-name">{t.name}</span>
              <span class="snapshot-topic-n">{t.keywords.length}</span>
            </button>
          ))}
        </div>
        {activeTopic && (
          <details class="snapshot-keywords">
            <summary>keywords · {activeTopic.keywords.length}</summary>
            <div class="snapshot-chip-cloud">
              {activeTopic.keywords.map((k) => (
                <span key={k} class="snapshot-chip">
                  {k}
                </span>
              ))}
            </div>
            <KeywordEditor topicId={activeTopic.id} />
          </details>
        )}
      </section>

      <section
        class={`snapshot-side-block snapshot-drop${props.dragOver ? " is-drag" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          props.setDragOver(true);
        }}
        onDragLeave={() => props.setDragOver(false)}
        onDrop={onDrop}
      >
        <h3>Load log</h3>
        <input
          ref={fileRef}
          type="file"
          accept=".json,.jsonl,application/json,text/plain"
          class="visually-hidden"
          onChange={async (e) => {
            const f = (e.currentTarget as HTMLInputElement).files?.[0];
            if (f) doParse(await f.text(), f.name);
          }}
        />
        <div class="snapshot-drop-actions">
          <button type="button" class="btn-ghost" onClick={() => fileRef.current?.click()}>
            Choose file
          </button>
          <button type="button" class="btn-ghost" onClick={() => loadSample()}>
            Sample
          </button>
        </div>
        <textarea
          class="snapshot-paste"
          placeholder="…or paste JSON / JSONL"
          value={pasteText.value}
          rows={5}
          onInput={(e) => (pasteText.value = (e.currentTarget as HTMLTextAreaElement).value)}
        />
        <input
          class="ctl-input"
          type="text"
          placeholder="log name (optional)"
          value={pasteName.value}
          onInput={(e) => (pasteName.value = (e.currentTarget as HTMLInputElement).value)}
        />
        <button
          type="button"
          class="btn-primary"
          disabled={!pasteText.value.trim()}
          onClick={() => doParse(pasteText.value, pasteName.value)}
        >
          Parse
        </button>
        {errorMsg.value && <p class="snapshot-error">{errorMsg.value}</p>}
      </section>

      {snap.logs.length > 0 && (
        <section class="snapshot-side-block">
          <h3>Logs</h3>
          <ul class="snapshot-log-list">
            {snap.logs.map((l) => (
              <li key={l.id} class={l.id === snap.activeLogId ? "is-active" : ""}>
                <button
                  type="button"
                  class="snapshot-log-btn"
                  onClick={() => appStore.getState().setActiveLog(l.id)}
                >
                  <span class="snapshot-log-name">{l.name}</span>
                  <span class="snapshot-log-meta">{l.turns.length} turns</span>
                </button>
                <button
                  type="button"
                  class="snapshot-log-x"
                  aria-label={`Remove ${l.name}`}
                  onClick={() => appStore.getState().removeSnapshotLog(l.id)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </aside>
  );
}

function KeywordEditor(props: { topicId: string }) {
  const draft = useSignal("");
  const snap = $snapshot.value;
  const topic = snap.topics.find((t) => t.id === props.topicId);
  if (!topic) return null;

  const add = () => {
    const k = draft.value.trim();
    if (!k || topic.keywords.includes(k)) return;
    appStore
      .getState()
      .updateTopicPreset(topic.id, { keywords: [...topic.keywords, k] });
    draft.value = "";
  };

  return (
    <div class="snapshot-keyword-edit">
      <input
        class="ctl-input"
        type="text"
        value={draft.value}
        placeholder="add keyword…"
        onInput={(e) => (draft.value = (e.currentTarget as HTMLInputElement).value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            add();
          }
        }}
      />
      <button type="button" class="btn-ghost" onClick={add} disabled={!draft.value.trim()}>
        add
      </button>
    </div>
  );
}

function SnapshotGraph(props: {
  keywords: string[];
  active: Set<string>;
  edges: { a: string; b: string; weight: number; lastTurn: number }[];
  totals: Map<string, number>;
}) {
  const size = 560;
  const cx = size / 2;
  const cy = size / 2;
  const layout = useMemo(
    () => layoutRadial(props.keywords, cx, cy, size * 0.36),
    [props.keywords.join("|"), cx, cy],
  );
  const maxWeight = Math.max(1, ...props.edges.map((e) => e.weight));
  const maxTotal = Math.max(1, ...props.totals.values());

  if (props.keywords.length === 0) {
    return (
      <div class="snapshot-graph is-empty">
        <p>Pick a topic on the left — the keyword ring will appear here.</p>
      </div>
    );
  }

  return (
    <svg class="snapshot-graph" viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Snapshot graph">
      <defs>
        <radialGradient id="node-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="var(--ramp-2)" stop-opacity="0.35" />
          <stop offset="100%" stop-color="var(--ramp-2)" stop-opacity="0" />
        </radialGradient>
      </defs>

      <circle
        class="snapshot-graph-ring"
        cx={cx}
        cy={cy}
        r={size * 0.36}
      />

      {props.edges.map((e) => {
        const a = layout.get(e.a);
        const b = layout.get(e.b);
        if (!a || !b) return null;
        const w = e.weight / maxWeight;
        return (
          <line
            key={`${e.a}::${e.b}`}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            class="snapshot-edge"
            stroke-width={0.6 + w * 3.4}
            stroke-opacity={0.15 + w * 0.6}
          />
        );
      })}

      {[...layout.entries()].map(([k, pos]) => {
        const total = props.totals.get(k) ?? 0;
        const isActive = props.active.has(k);
        const r = isActive ? 6 + (total / maxTotal) * 8 : 4;
        return (
          <g
            key={k}
            class={`snapshot-node${isActive ? " is-active" : ""}`}
            transform={`translate(${pos.x} ${pos.y})`}
          >
            {isActive && <circle class="snapshot-node-glow" r={r + 10} fill="url(#node-glow)" />}
            <circle class="snapshot-node-core" r={r} />
            <text
              class="snapshot-node-label"
              x={0}
              y={r + 14}
              text-anchor="middle"
              dominant-baseline="hanging"
            >
              {k}
            </text>
            {isActive && (
              <text
                class="snapshot-node-count"
                x={0}
                y={-r - 6}
                text-anchor="middle"
              >
                {total}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function SnapshotScrubber(props: {
  disabled: boolean;
  value: number;
  max: number;
  onChange: (i: number) => void;
}) {
  const snap = $snapshot.value;
  return (
    <div class={`snapshot-scrubber${props.disabled ? " is-disabled" : ""}`}>
      <button
        type="button"
        class="btn-ghost"
        disabled={props.disabled || props.value === 0}
        onClick={() => props.onChange(Math.max(0, props.value - 1))}
        aria-label="Previous turn"
      >
        ‹
      </button>
      <button
        type="button"
        class="btn-ghost"
        disabled={props.disabled}
        onClick={() => appStore.getState().setPlaying(!snap.playing)}
      >
        {snap.playing ? "◼ stop" : "▶ play"}
      </button>
      <input
        type="range"
        class="ctl-slider snapshot-range"
        min={0}
        max={props.max}
        value={props.value}
        disabled={props.disabled}
        onInput={(e) =>
          props.onChange(Number((e.currentTarget as HTMLInputElement).value))
        }
      />
      <span class="snapshot-scrubber-pos">
        {props.disabled ? "—" : `${props.value + 1} / ${props.max + 1}`}
      </span>
      <button
        type="button"
        class="btn-ghost"
        disabled={props.disabled || props.value === props.max}
        onClick={() => props.onChange(Math.min(props.max, props.value + 1))}
        aria-label="Next turn"
      >
        ›
      </button>
    </div>
  );
}

function loadSample() {
  const log = parseConversationText(SAMPLE_LOG_TEXT, "sample conversation");
  appStore.getState().addSnapshotLog(log);
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1).trimEnd() + "…";
}
