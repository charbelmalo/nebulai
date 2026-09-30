/** MapChooser.tsx — Research's Comparisons page before a map is chosen.
 *
 *  Research never opens the SAE starter on its own (a starter unit is not a
 *  GPT-2 measurement), so its map page starts here: every published export,
 *  with the facts that decide which advanced modes it supports, and one
 *  button each. Picking one is the explicit request; the Type list then
 *  offers Chord, Hierarchical and Compare with their own requirements.
 *
 *  Each card names the map in words (model · what a point is · layer) with
 *  the raw export id as its caption, groups by model family, and shows the
 *  map's own thumbnail. Search filters on all three. */

import { signal } from "@preact/signals";
import { requestDataset } from "../app/actions";
import type { DatasetEntry } from "../data/schema";
import { describeDataset, type DatasetCard } from "../data/representation";
import { DatasetThumb } from "./DatasetThumb";
import { $datasets, $loadError, $pendingDatasetId } from "./state";

const FAMILY_ORDER = ["GPT-2 family", "SmolLM2", "Other models", "Concept probes"];
/** a group longer than this shows its first cards and a "Show all" button */
const GROUP_CAP = 6;

const $query = signal("");
const $expanded = signal<ReadonlySet<string>>(new Set());

function matches(d: DatasetEntry, c: DatasetCard, q: string): boolean {
  if (!q) return true;
  const hay = `${c.title} ${c.subtitle} ${d.id} ${d.model}`.toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((w) => hay.includes(w));
}

export function MapChooser() {
  const datasets = $datasets.value;
  const pending = $pendingDatasetId.value;
  const error = $loadError.value;
  const q = $query.value.trim();
  const cards = datasets.map((d) => ({ d, c: describeDataset(d) }));
  const shown = cards.filter(({ d, c }) => matches(d, c, q));
  const groups = FAMILY_ORDER.map((family) => ({
    family,
    items: shown.filter(({ c }) => c.family === family),
  })).filter((g) => g.items.length > 0);

  return (
    <section class="map-chooser" aria-labelledby="map-chooser-title">
      <header class="map-chooser-head">
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
      </header>
      {error && (
        <p class="map-chooser-error" role="alert">
          {error.message}
        </p>
      )}
      {datasets.length === 0 ? (
        <p class="map-chooser-empty">No model maps are published here.</p>
      ) : (
        <>
          <div class="map-chooser-tools">
            <label class="map-chooser-search">
              <span class="sr-only">Filter maps</span>
              <input
                type="search"
                placeholder="Filter by model, unit or layer…"
                value={$query.value}
                onInput={(e) => ($query.value = (e.target as HTMLInputElement).value)}
              />
            </label>
            <p class="map-chooser-count" role="status">
              {q ? `${shown.length} of ${datasets.length} maps` : `${datasets.length} maps`}
            </p>
          </div>
          {groups.length === 0 && (
            <p class="map-chooser-empty">
              No map matches “{q}”.{" "}
              <button type="button" class="map-chooser-link" onClick={() => ($query.value = "")}>
                Clear the filter
              </button>
            </p>
          )}
          {groups.map(({ family, items }) => {
            const open = q !== "" || $expanded.value.has(family) || items.length <= GROUP_CAP;
            const visible = open ? items : items.slice(0, GROUP_CAP);
            const gid = `map-group-${family.replace(/\W+/g, "-").toLowerCase()}`;
            return (
              <section key={family} class="map-chooser-group" aria-labelledby={gid}>
                <h2 id={gid} class="map-chooser-group-title">
                  {family} <span>{items.length}</span>
                </h2>
                <ul class="map-chooser-list">
                  {visible.map(({ d, c }) => (
                    <li key={d.id}>
                      <button
                        type="button"
                        class="map-chooser-item"
                        disabled={pending !== null}
                        aria-busy={pending === d.id}
                        onClick={() => requestDataset(d.id)}
                      >
                        <DatasetThumb path={d.path} />
                        <span class="map-chooser-body">
                          <span class="map-chooser-name">{c.title}</span>
                          <span class="map-chooser-sub">{c.subtitle}</span>
                          <span class="map-chooser-meta">
                            {d.n_points.toLocaleString("en-US")} points · {d.n_clusters} clusters
                            {d.has_edges ? " · hierarchy" : ""}
                          </span>
                          <span class="map-chooser-id">{d.id}</span>
                          {pending === d.id && <span class="map-chooser-state">Opening…</span>}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
                {!open && (
                  <button
                    type="button"
                    class="map-chooser-more"
                    onClick={() => ($expanded.value = new Set([...$expanded.value, family]))}
                  >
                    Show all {items.length} {family} maps
                  </button>
                )}
              </section>
            );
          })}
        </>
      )}
    </section>
  );
}
