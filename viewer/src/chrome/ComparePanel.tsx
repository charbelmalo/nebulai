/** Compare-mode side panel — the model legend, layout-state switcher, and
 *  honesty stats (shared/unique counts, Jaccard overlaps) for the cross-model
 *  view. Replaces LegendCard while viewMode === "compare"; state lives in the
 *  store's compare slice so the CompareDriver mirrors it without touching
 *  Preact. */

import { appStore } from "../app/store";
import { $compare, $compareData } from "./state";
import { RadioRow, ToggleRow } from "./controls";

const STATE_LABELS: Record<string, string> = {
  native: "Native clouds",
  semantic: "Semantic space",
  by_model: "By model",
  by_concept: "By concept",
};

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
        value={data.states[ui.state] ?? "semantic"}
        options={data.states.map((s) => ({ value: s, label: STATE_LABELS[s] ?? s }))}
        onChange={(v) => st.setCompareState(Math.max(data.states.indexOf(v), 0))}
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
