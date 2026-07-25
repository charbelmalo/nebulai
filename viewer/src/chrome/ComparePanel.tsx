/** Compare-mode side panel — the model legend, layout-state switcher, and
 *  honesty stats (shared/unique counts, Jaccard overlaps) for the cross-model
 *  view. Replaces LegendCard while viewMode === "compare"; state lives in the
 *  store's compare slice so the CompareDriver mirrors it without touching
 *  Preact. */

import { requestCompareTour } from "../app/actions";
import { appStore } from "../app/store";
import { $compare, $compareData, $compareTour } from "./state";
import { RadioRow, ToggleRow } from "./controls";

const STATE_LABELS: Record<string, string> = {
  native: "Native clouds",
  semantic: "Semantic space",
  by_model: "By model",
  by_concept: "By concept",
};

const label = (s: string | undefined) => (s ? (STATE_LABELS[s] ?? s) : "—");

export function ComparePanel() {
  const data = $compareData.value;
  const ui = $compare.value;
  if (!data) return null;

  const st = appStore.getState();

  return (
    <section class="legend compare-panel" aria-label="Comparison">
      <header class="legend-head">
        <h2 class="legend-title">Model comparison</h2>
      </header>
      <p class="legend-caption">
        {data.meta.n_points} cluster concepts from {data.meta.models.length} models · embedded in{" "}
        {data.meta.embed_model} — label space, not model geometry
      </p>

      <RadioRow
        name="Layout"
        // the DRIVER's stage, not the store's — the tour moves the field
        // without writing the store, and a radio that disagreed with what is on
        // screen would be the panel lying about the view
        value={$compareTour.value?.stageName || data.states[ui.state] || "semantic"}
        options={data.states.map((s) => ({ value: s, label: STATE_LABELS[s] ?? s }))}
        onChange={(v) => {
          const i = Math.max(data.states.indexOf(v), 0);
          // the driver first: picking the layout the store already holds is not
          // a store change, so this is the only signal it would get
          requestCompareTour({ kind: "pick", state: i });
          st.setCompareState(i);
        }}
      />

      <div class="legend-sep" />
      <div class="compare-models" role="group" aria-label="Models">
        {data.meta.models.map((m, i) => {
          const c = data.colors[m] ?? [0.6, 0.6, 0.6];
          const n = data.points.reduce((acc, p) => acc + (p.source_idx === i ? 1 : 0), 0);
          const off = ui.hiddenModels.includes(i);
          return (
            <button
              key={m}
              type="button"
              class={`compare-model${off ? " is-off" : ""}`}
              aria-pressed={!off}
              onClick={() => st.toggleCompareModel(i)}
            >
              <span
                class="compare-swatch"
                style={{
                  background: `rgb(${Math.round(c[0] * 255)} ${Math.round(c[1] * 255)} ${Math.round(c[2] * 255)})`,
                }}
              />
              <span class="compare-model-name">{m}</span>
              <span class="compare-model-n">{n}</span>
            </button>
          );
        })}
      </div>
      <ToggleRow
        label="Shared concepts only"
        checked={ui.sharedOnly}
        onChange={(v) => st.setCompareSharedOnly(v)}
      />

      <div class="legend-sep" />
      <dl class="compare-stats">
        <div class="compare-stat">
          <dt>shared concepts</dt>
          <dd>{data.stats.n_shared_concepts}</dd>
        </div>
        {Object.entries(data.stats.n_unique_per_model).map(([m, n]) => (
          <div key={m} class="compare-stat">
            <dt>unique · {m}</dt>
            <dd>{n}</dd>
          </div>
        ))}
      </dl>
      <p class="legend-caption">concept overlap (Jaccard)</p>
      <dl class="compare-stats">
        {Object.entries(data.stats.jaccard).map(([k, v]) => (
          <div key={k} class="compare-stat">
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

/** The layout tour — play the four layouts start to finish.
 *
 *  HONESTY: this is a transport, not a timeline. Compare has no time axis, so
 *  the readout names layouts and counts stages; there is deliberately no clock
 *  anywhere in it, because seconds here would be a property of the animation,
 *  not of the data. The scrub ticks come from the driver's own stop positions
 *  so a tick always lands where that layout is actually settled. */
export function CompareTransport() {
  const t = $compareTour.value;
  const data = $compareData.value;
  if (!t || !data || !t.hasData) return null;

  const morphing = t.blend > 0;
  const next = data.states[t.stage + 1];
  // ticks mark where each layout sits along the tour
  const ticks = t.stops
    .map((u) => {
      const p = (u * 100).toFixed(2);
      return `transparent calc(${p}% - 1px), var(--tp-tick) calc(${p}% - 1px), var(--tp-tick) calc(${p}% + 1px), transparent calc(${p}% + 1px)`;
    })
    .join(", ");

  return (
    <div class="transport compare-transport" role="group" aria-label="Layout tour">
      <button
        type="button"
        class="tp-btn tp-play"
        aria-label={t.playing ? "Pause layout tour" : "Play layout tour"}
        onClick={() => requestCompareTour({ kind: "toggle" })}
      >
        {t.playing ? "⏸" : "▶"}
      </button>
      <button
        type="button"
        class="tp-btn"
        aria-label="Restart tour from the first layout"
        title="Restart"
        onClick={() => requestCompareTour({ kind: "restart" })}
      >
        ↺
      </button>
      <input
        class="tp-scrub tp-scrub-ticked"
        type="range"
        min={0}
        max={1}
        step={0.001}
        value={t.progress}
        aria-label="Tour position"
        aria-valuetext={
          morphing
            ? `morphing from ${label(t.stageName)} to ${label(next)}`
            : `${label(t.stageName)}, layout ${t.stage + 1} of ${t.stages}`
        }
        // scrubbing pauses first, so the drag never fights the tour
        onInput={(e) =>
          requestCompareTour({ kind: "seek", u: Number((e.currentTarget as HTMLInputElement).value) })
        }
        style={{
          "--tp-progress": `${(t.progress * 100).toFixed(1)}%`,
          "--tp-ticks": `linear-gradient(to right, ${ticks})`,
        }}
      />
      <span class="tp-stage" aria-hidden="true">
        <span class="tp-now">{label(t.stageName)}</span>
        {morphing && <span class="tp-sep">→</span>}
        {morphing && <span class="tp-total">{label(next)}</span>}
      </span>
      <span class="tp-count" title="layout position in the tour">
        {t.stage + 1}
        <i>/{t.stages}</i>
      </span>
      <label class="tp-speed">
        <span class="visually-hidden">Tour speed</span>
        <select
          value={String(t.speed)}
          onChange={(e) =>
            requestCompareTour({
              kind: "speed",
              mult: Number((e.currentTarget as HTMLSelectElement).value),
            })
          }
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
