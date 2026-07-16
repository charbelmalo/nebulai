/** Keyword search over the token labels (atlas view). Typing dims the map to
 *  the match set; results group by cluster so a query doubles as a pattern
 *  check ("where does the model put ship-words?"). Matching is honest
 *  case-insensitive substring — no fuzzy ranking that would fake semantics. */

import { useSignal } from "@preact/signals";
import { requestFlyToCluster, requestFlyToPoint } from "../app/actions";
import { appStore } from "../app/store";
import { knnNeighbors } from "../data/edges";
import { $dataset, $mapQuery, $selection } from "./state";

/** hard cap on rendered rows across all groups — 50K-token vocabularies can
 *  match thousands of rows and the panel must stay a panel, not a dump */
const MAX_ROWS = 100;
const ROWS_PER_GROUP = 8;
const KNN_CHIPS = 5;

export function SearchPanel() {
  const collapsed = useSignal(false);
  const ds = $dataset.value;
  if (!ds) return null;

  const { text, results } = $mapQuery.value;
  const setQuery = (t: string) => appStore.getState().setMapQuery(t);

  if (collapsed.value) {
    return (
      <button
        type="button"
        class="search-fab"
        aria-label="Open search"
        onClick={() => (collapsed.value = false)}
      >
        Search
      </button>
    );
  }

  // groups sorted by match count desc; -1 (noise) titled honestly
  const clustersById = new Map(ds.columns.clusters.map((c) => [c.id, c]));
  const groups = results
    ? [...results.byCluster.entries()].sort((a, b) => b[1].length - a[1].length)
    : [];

  let rowBudget = MAX_ROWS;

  return (
    <section class="search-panel" aria-label="Keyword search">
      <header class="search-head">
        <h2 class="search-title">Search</h2>
        {results && (
          <span class="search-count">
            {results.total.toLocaleString()} of {ds.columns.labels.length.toLocaleString()}
          </span>
        )}
        <button
          type="button"
          class="legend-collapse"
          aria-label="Collapse search"
          onClick={() => (collapsed.value = true)}
        >
          ›
        </button>
      </header>

      <input
        class="search-input"
        type="search"
        placeholder="find tokens… (substring, case-insensitive)"
        value={text}
        onInput={(e) => setQuery((e.currentTarget as HTMLInputElement).value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") setQuery("");
        }}
      />

      {results && results.total === 0 && (
        <p class="search-empty">
          no token contains “{text.trim()}” in this vocabulary
        </p>
      )}

      {results && results.total > 0 && (
        <div class="search-groups">
          {groups.map(([cid, ids]) => {
            if (rowBudget <= 0) return null;
            const shown = ids.slice(0, Math.min(ROWS_PER_GROUP, rowBudget));
            rowBudget -= shown.length;
            const title = cid < 0 ? "noise (unclustered)" : clustersById.get(cid)?.title ?? `cluster ${cid}`;
            return (
              <div class="search-group" key={cid}>
                <button
                  type="button"
                  class="search-group-head"
                  title={cid < 0 ? undefined : "select cluster + fly to it"}
                  onClick={() => {
                    if (cid < 0) return;
                    appStore.getState().setSelection({ kind: "cluster", id: cid });
                    requestFlyToCluster(cid);
                  }}
                >
                  <span class="search-group-title">{title}</span>
                  <span class="search-group-count">{ids.length.toLocaleString()}</span>
                </button>
                {shown.map((id) => (
                  <SearchRow key={id} id={id} />
                ))}
                {ids.length > shown.length && (
                  <p class="search-more">…and {(ids.length - shown.length).toLocaleString()} more</p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

/** One match row; when selected it expands with its kNN row — the "links in
 *  thinking patterns" affordance (sims are 10-D cluster-space, per legend). */
function SearchRow({ id }: { id: number }) {
  const ds = $dataset.value!;
  const sel = $selection.value;
  const selected = sel?.kind === "point" && sel.id === id;
  const edges = ds.columns.edges;
  const neighbors = selected && edges ? knnNeighbors(edges, id).slice(0, KNN_CHIPS) : [];

  return (
    <div class={selected ? "search-row is-selected" : "search-row"}>
      <button
        type="button"
        class="search-row-label"
        onClick={() => {
          appStore.getState().setSelection({ kind: "point", id });
          requestFlyToPoint(id);
        }}
      >
        {ds.columns.labels[id]}
      </button>
      {selected && neighbors.length > 0 && (
        <div class="search-chips">
          {neighbors.map((nb) => (
            <button
              type="button"
              class="search-chip"
              key={nb.id}
              title={`similarity ${nb.sim.toFixed(3)} (10-D cluster space)`}
              onClick={() => {
                appStore.getState().setSelection({ kind: "point", id: nb.id });
                requestFlyToPoint(nb.id);
              }}
            >
              {ds.columns.labels[nb.id]}
              <span class="search-chip-sim">{nb.sim.toFixed(2)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
