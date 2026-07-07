/** Bottom-right collapsible legend — the video's "Connections" card. Shows
 *  the weight ramp with honest provenance (metric @ space), the view-mode
 *  radio group, and a truthful empty state for v1 exports without edges. */

import { useSignal } from "@preact/signals";
import { requestViewMode } from "../app/actions";
import type { ViewMode } from "../app/store";
import { $dataset, $toggles, $viewMode } from "./state";
import { RadioRow } from "./controls";

export function LegendCard() {
  const collapsed = useSignal(false);
  if (!$toggles.value.legend) return null;

  const edges = $dataset.value?.columns.edges ?? null;

  if (collapsed.value) {
    return (
      <button
        type="button"
        class="legend-fab"
        aria-label="Open legend"
        onClick={() => (collapsed.value = false)}
      >
        Connections
      </button>
    );
  }

  return (
    <section class="legend" aria-label="Legend">
      <header class="legend-head">
        <h2 class="legend-title">Connections</h2>
        <button
          type="button"
          class="legend-collapse"
          aria-label="Collapse legend"
          onClick={() => (collapsed.value = true)}
        >
          ›
        </button>
      </header>

      {edges ? (
        <>
          <div class="legend-ramp" aria-hidden="true" />
          <div class="legend-ticks" aria-hidden="true">
            <span>0</span>
            <span>0.5</span>
            <span>1</span>
          </div>
          <p class="legend-caption">
            {edges.metric.replace(/_/g, " ")} similarity in {edges.space} — 10-D cluster
            space, not screen distance
          </p>
        </>
      ) : (
        <p class="legend-empty">
          no edges in this export (schema v1) — re-export with{" "}
          <code>nebulai edges &lt;model&gt;</code> to light up connections
        </p>
      )}

      <div class="legend-sep" />
      <RadioRow
        name="View"
        value={$viewMode.value}
        options={[
          { value: "atlas", label: "Atlas" },
          { value: "chord", label: "Chord" },
          {
            value: "hierarchy",
            label: "Hierarchical network",
            disabled: !edges,
            hint: !edges ? "needs edges (v2 export)" : undefined,
          },
        ]}
        onChange={(v) => requestViewMode(v as ViewMode)}
      />
    </section>
  );
}
