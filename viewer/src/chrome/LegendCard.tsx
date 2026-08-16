/** Bottom-right collapsible legend — the video's "Connections" card. Shows
 *  the weight ramp with honest provenance (metric @ space), what the LAYOUT
 *  means for the active view, the map's measured distortion, the view-mode
 *  radio group, and a truthful empty state for v1 exports without edges.
 *
 *  The layout note and the distortion readout close two gaps that stood for as
 *  long as the viewer has existed: the beams carried their provenance while the
 *  positions carried none — the word "UMAP" appeared nowhere in the map UI —
 *  and trustworthiness / seed-ARI / null-margin stopped at the CLI, so a
 *  browser user could not tell a faithful map from one whose separation the
 *  null out-scores. Both are read, never recomputed; see data/validation.ts. */

import { requestViewMode } from "../app/actions";
import type { ViewMode } from "../app/store";
import { ensureVerdicts, verdictFor } from "../data/validation";
import {
  $dataset,
  $datasetId,
  $legendCollapsed,
  $toggles,
  $viewMode,
  openPanel,
} from "./state";
import { RadioRow } from "@psychix/viz/controls";

/** What a POSITION means in each view. Every clause here is a statement the
 *  drivers already make in their headers — ChordDriver rim order comes from the
 *  atlas centroid bearing but is rank-spaced, HierarchyDriver's radius is the
 *  merge weight and its angle is DFS leaf order — surfaced where the reader is
 *  rather than where the author was. */
/** `meta.unit` is `kind` or `kind(source, module_path)`. Only the kind is prose;
 *  everything in the parens is a repo id or a module path and stays VERBATIM —
 *  the edges caption's blanket `_`→space would turn `blocks.8.hook_resid_pre`
 *  into "hook resid pre", which is not a thing anyone can look up. */
function unitLabel(unit: string | null): string {
  if (!unit) return "unit";
  const i = unit.indexOf("(");
  return i < 0
    ? unit.replace(/_/g, " ")
    : `${unit.slice(0, i).replace(/_/g, " ")} ${unit.slice(i)}`;
}

function layoutNote(view: ViewMode, unit: string | null): string {
  const what = unitLabel(unit);
  switch (view) {
    case "chord":
      return `Rim order follows the atlas layout, but the rim is spaced evenly by rank — the gap between two nodes is position in the ring, not similarity. Chord weight is the similarity.`;
    case "hierarchy":
      return `Radius is the similarity two subtrees merge at, normalized to the strongest merge. Angle is leaf order from the tree walk, not similarity — neighbouring leaves are adjacent in the ordering.`;
    default:
      return `UMAP projection of the ${what} vectors. Axes carry no units and no direction — only which points sit near each other is a claim. The distance between two clusters is not.`;
  }
}

function num(v: number | null | undefined, places = 4): string | null {
  return typeof v === "number" ? v.toFixed(places) : null;
}

/** trustworthiness / seed stability / null margin, exactly as `nebulai metrics`
 *  ruled on them. Renders nothing at all when the map was never validated —
 *  absence is "not measured", and a blank readout is honest where a passing one
 *  would not be. */
function DistortionReadout() {
  ensureVerdicts();
  const v = verdictFor($datasetId.value);
  if (!v) return null;

  const trust = num(v.trustworthiness);
  const ari = num(v.stability_ari);
  const margin = typeof v.margin === "number" ? v.margin : null;
  if (trust === null && ari === null && margin === null) return null;

  const flag = v.below_null_floor ? "!" : v.not_comparable ? "?" : "";

  return (
    <div class="legend-validation">
      <p class="legend-caption legend-validation-row">
        {trust !== null && (
          <span title="neighbourhood preservation from the original space (1.0 faithful, ~0.5 chance)">
            trust <b>{trust}</b>
          </span>
        )}
        {ari !== null && (
          <span title="mean pairwise agreement across UMAP seeds (1.0 = same map every seed)">
            seed ARI <b>{ari}</b>
          </span>
        )}
        {margin !== null && (
          <span
            class={v.below_null_floor ? "legend-below-null" : undefined}
            title="silhouette minus the silhouette of the same pipeline on column-shuffled vectors"
          >
            null margin{" "}
            <b>
              {margin >= 0 ? "+" : ""}
              {margin.toFixed(4)}
              {flag}
            </b>
          </span>
        )}
      </p>
      {v.below_null_floor && (
        <p class="legend-caption legend-below-null">
          The null out-scored this map — the same pipeline found as much
          separation in column-shuffled vectors. Read the clusters here as a
          property of the construction, not the model. Trust still holds: it
          never touches HDBSCAN.
        </p>
      )}
      {!v.below_null_floor && v.not_comparable && (
        <p class="legend-caption">
          The null resolved a very different cluster count, so the margin compares
          two different questions and is not evidence either way.
        </p>
      )}
    </div>
  );
}

export function LegendCard() {
  if (!$toggles.value.legend) return null;

  const edges = $dataset.value?.columns.edges ?? null;

  if ($legendCollapsed.value) {
    return (
      <button
        type="button"
        class="legend-fab"
        aria-label="Open legend"
        onClick={() => openPanel("legend")}
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
          onClick={() => ($legendCollapsed.value = true)}
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

      {/* LAST, and directly under the view switcher, for two reasons. The note
          describes the SELECTED view, so it reads as an answer to the control
          above it. And it is the only part of this card whose length varies —
          a below-null callout adds several lines — so keeping it at the bottom
          means the growth pushes prose past the card's scroll bound, never the
          ramp or the radio group. */}
      <div class="legend-sep" />
      <h3 class="legend-subtitle">Positions</h3>
      <p class="legend-caption">
        {layoutNote($viewMode.value, $dataset.value?.columns.meta.unit ?? null)}
      </p>
      <DistortionReadout />
    </section>
  );
}
