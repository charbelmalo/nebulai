/** Keyword search over the token labels (atlas view). Typing dims the map to
 *  the match set; results group by cluster so a query doubles as a pattern
 *  check ("where does the model put ship-words?"). Matching is honest
 *  case-insensitive substring — no fuzzy ranking that would fake semantics. */

import { useSignal } from "@preact/signals";
import { requestFlyToCluster, requestFlyToPoint } from "../app/actions";
import { appStore } from "../app/store";
import { knnDistance, knnDistanceFloor, knnNeighbors } from "../data/edges";
import { rampRgb } from "../scene/interp/chart-theme";
import { $dataset, $mapQuery, $selection } from "./state";

/** hard cap on rendered rows across all groups — 50K-token vocabularies can
 *  match thousands of rows and the panel must stay a panel, not a dump */
const MAX_ROWS = 100;
const ROWS_PER_GROUP = 8;

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

/** One match row; when selected it expands into its full ranked kNN table. */
function SearchRow({ id }: { id: number }) {
  const ds = $dataset.value!;
  const sel = $selection.value;
  const selected = sel?.kind === "point" && sel.id === id;

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
      {selected && <KnnTable id={id} />}
    </div>
  );
}

/** The ranked nearest-neighbour table — every neighbour the export stored for
 *  this point, in rank order, with its score, the distance behind that score,
 *  and a bar for scanning the falloff.
 *
 *  On the two things this deliberately does NOT have:
 *
 *  · **No metric selector.** The audit asked for one. There is nothing to
 *    select between. Neighbours were ranked ONCE, offline, by exact Euclidean
 *    distance in 10-D `u_cluster` space, and only the resulting ids and scores
 *    were exported — the source vectors are not in the bundle. A selector
 *    would either do nothing or re-sort a fixed list under a metric this build
 *    cannot compute, which is a control that lies about what it changes.
 *  · **No "similarity" label.** The stored number is `exp(-(d/sigma)^2)`, a
 *    unitless kernel value, not a cosine. Calling it similarity invites reading
 *    0.85 as an angle. It is headed `score`, and the distance it came from sits
 *    next to it.
 *
 *  The BAR draws the distance, not the score, and that choice came from the
 *  data rather than from taste. The kernel saturates hard: sigma is the median
 *  neighbour distance over the whole export, so on the bundled GPT-2 atlas a
 *  typical point's own nearest neighbour already sits past 2*sigma, scoring
 *  0.006, and the rest of its row rounds to 0.000. Drawn on a 0–1 score scale
 *  every bar in this panel would be an empty track — technically honest and
 *  completely useless. The distances behind those same scores (0.17, 0.19, …)
 *  are well spread and are what the neighbour search actually ranked on. */
function KnnTable({ id }: { id: number }) {
  const ds = $dataset.value!;
  const edges = ds.columns.edges;
  const sigma = edges?.knn?.sigma ?? 0;
  const neighbors = edges ? knnNeighbors(edges, id) : [];
  if (neighbors.length === 0) return null;

  // Fixed track: 0 to the distance at which scores round away. Fixed, not
  // fitted to this row set — a bar has to mean the same thing after clicking
  // through to another point, and the floor is a property of the export, so
  // it is the one honest bound available.
  const floor = knnDistanceFloor(sigma);

  return (
    <div class="knn-table">
      <div class="knn-head">
        <span class="knn-head-title">nearest neighbours</span>
        <span
          class="knn-head-meta"
          title={
            `Ranked offline by exact Euclidean distance in 10-D u_cluster space. ` +
            `score = exp(-(d/sigma)^2), sigma = ${sigma.toFixed(4)} (the median ` +
            `neighbour distance across the whole export, so scores and distances ` +
            `are comparable between points). Bars show distance on a fixed ` +
            `0–${floor.toFixed(2)} track; ${floor.toFixed(2)} is where a score ` +
            `rounds to 0.000 and the distance stops being recoverable.`
          }
        >
          k={neighbors.length} · 10-D u_cluster · σ={sigma.toFixed(3)}
        </span>
      </div>
      <ol class="knn-rows">
        {neighbors.map((nb, i) => {
          const d = knnDistance(nb.sim, sigma);
          // Colour agrees with length: both read d, so near is ramp-hot and
          // far is ramp-cold. `1 - d/floor` because the ramp's low end is the
          // attention-getting one and NEAR is what deserves it.
          const t = d === null ? 0 : Math.max(0, Math.min(1, 1 - d / floor));
          const [r, g, b] = rampRgb(t);
          return (
            <li class="knn-row" key={nb.id}>
              <span class="knn-rank">{i + 1}</span>
              <button
                type="button"
                class="knn-label"
                onClick={() => {
                  appStore.getState().setSelection({ kind: "point", id: nb.id });
                  requestFlyToPoint(nb.id);
                }}
              >
                {ds.columns.labels[nb.id]}
              </button>
              <span class="knn-bar" aria-hidden="true">
                <span
                  class="knn-bar-fill"
                  style={{
                    width: `${(d === null ? 1 : Math.min(1, d / floor)) * 100}%`,
                    background: d === null ? "var(--hairline)" : `rgb(${r},${g},${b})`,
                  }}
                />
              </span>
              <span class="knn-score">{nb.sim.toFixed(3)}</span>
              <span
                class="knn-dist"
                title={
                  d === null
                    ? `Farther than ${floor.toFixed(2)}. The exact distance is gone: ` +
                      `scores are rounded to 3 decimals on export and this one rounded ` +
                      `to 0.000, so only the lower bound survives.`
                    : "Euclidean distance in 10-D u_cluster space"
                }
              >
                {d === null ? `>${floor.toFixed(1)}` : d.toFixed(2)}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
