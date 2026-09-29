/** MapChooser.tsx — Research's Comparisons page before a map is chosen.
 *
 *  Research never opens the SAE starter on its own (a starter unit is not a
 *  GPT-2 measurement), so its map page starts here: every published export,
 *  with the facts that decide which advanced modes it supports, and one
 *  button each. Picking one is the explicit request; the Type list then
 *  offers Chord, Hierarchical and Compare with their own requirements. */

import { requestDataset } from "../app/actions";
import { $datasets, $loadError, $pendingDatasetId } from "./state";

export function MapChooser() {
  const datasets = $datasets.value;
  const pending = $pendingDatasetId.value;
  const error = $loadError.value;
  return (
    <section class="map-chooser" aria-labelledby="map-chooser-title">
      <p class="map-chooser-kicker">Comparisons</p>
      <h1 id="map-chooser-title" class="map-chooser-title">
        Choose a model map
      </h1>
      <p class="map-chooser-lede">
        Research opens a map only when you ask for one. Pick an export below; Chord,
        Hierarchical and Compare views then appear under Type when that export supports
        them. These views arrange labels and directions — they are not activations measured
        on your input.
      </p>
      {error && (
        <p class="map-chooser-error" role="alert">
          {error.message}
        </p>
      )}
      {datasets.length === 0 ? (
        <p class="map-chooser-empty">No model maps are published here.</p>
      ) : (
        <ul class="map-chooser-list">
          {datasets.map((d) => (
            <li key={d.id}>
              <button
                type="button"
                class="map-chooser-item"
                disabled={pending !== null}
                aria-busy={pending === d.id}
                onClick={() => requestDataset(d.id)}
              >
                <span class="map-chooser-id">{d.id}</span>
                <span class="map-chooser-meta">
                  {d.model} · {d.n_points.toLocaleString()} points · {d.n_clusters} clusters
                  {d.has_edges ? " · hierarchy available" : ""}
                </span>
                {pending === d.id && <span class="map-chooser-state">Opening…</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
