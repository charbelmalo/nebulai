/** AtlasWorkspace — Atlas's left panel: what map this is (evidence header),
 *  the map controls that matter for the first task, and the results table
 *  that is the keyboard and no-GPU route to every unit.
 *
 *  The table is not a second search implementation: it pages through the same
 *  `mapQuery.results` the map dims to, and "Inspect" writes the same
 *  selection a map click writes. So a unit reached from the list and a unit
 *  picked on the canvas open the identical inspector with identical evidence.
 *
 *  Row order is the artifact's own row order, so point 0 is always the first
 *  row of the unfiltered list. */

import { signal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";
import {
  requestDataset,
  requestFlyToPoint,
  requestOpenPinned,
  requestRetryLoad,
  requestStarter,
} from "../../app/actions";
import { appStore, type Toggles } from "../../app/store";
import { activeManifest, currentArtifact } from "../../data/experience";
import { findingPin, parseFindingText, FINDING_MAX_BYTES } from "../../data/finding";
import { sourceUnit } from "../../data/columns";
import type { Dataset } from "../../data/loader";
import { parseCuration, REPRESENTATION_COPY } from "../../data/representation";
import { SelectRow, SliderRow, ToggleRow } from "@psychix/viz/controls";
import { handControlUnavailableReason } from "../../hands/types";
import { ChannelLens, DirectionMaker } from "../SearchPanel";
import { advancedHref } from "../Sidebar";
import {
  $dataset,
  $datasetId,
  $datasets,
  $dims,
  $loadError,
  $loading,
  $mapQuery,
  $pendingDatasetId,
  $pin,
  $renderer,
  $selection,
  $settings,
  $toggles,
  $unverifiedDefault,
} from "../state";
import {
  clusterTitle,
  countLine,
  fmtInt,
  isUnlabelled,
  labelSource,
  mapTitle,
  metaOf,
  modelLine,
  pct,
  representationOf,
  shortSha,
  str,
  unitTitle,
} from "./evidence";

export const PAGE_SIZE = 50;

/** results page, reset whenever the list underneath changes */
const $page = signal(0);
appStore.subscribe((s, prev) => {
  if (s.mapQuery !== prev.mapQuery || s.datasetId !== prev.datasetId) $page.value = 0;
});

/** The element that opened the inspector, so closing it can hand focus back
 *  to the row the reader came from. */
let returnFocus: HTMLElement | null = null;
export function rememberReturnFocus(el: HTMLElement | null): void {
  returnFocus = el;
}
export function restoreFocus(): void {
  const el = returnFocus;
  returnFocus = null;
  if (el && el.isConnected) {
    el.focus();
    return;
  }
  document.querySelector<HTMLElement>(".rt-search")?.focus();
}

/** Select a row, open its inspector and bring it into view. The same path
 *  for a results click, a pinned open and the Learn lesson. */
export function inspectRow(row: number, from?: HTMLElement | null): void {
  rememberReturnFocus(from ?? null);
  const st = appStore.getState();
  st.setSelection({ kind: "point", id: row });
  st.setInspectorOpen(true);
  requestFlyToPoint(row);
}

/* ── evidence header ─────────────────────────────────────────────────── */

export function EvidenceHeader({ ds, datasetId }: { ds: Dataset; datasetId: string }) {
  const meta = metaOf(ds);
  const rep = representationOf(ds);
  const copy = REPRESENTATION_COPY[rep];
  const cur = parseCuration(meta.curation);
  const noise = typeof meta.noise_fraction === "number" ? meta.noise_fraction : null;
  const current = currentArtifact(activeManifest(), datasetId);
  const earlier = !!(ds.sha256 && current && current.sha256 !== ds.sha256);
  return (
    <header class="aw-evidence">
      <h2 class="aw-title">{mapTitle(ds, datasetId)}</h2>
      <p class="aw-facts">
        {countLine(ds)} · Labels: {labelSource(meta)}
      </p>
      <p class="aw-meaning">
        Each point is one {copy.pointNoun}. {copy.limit}
      </p>
      {$unverifiedDefault.value && (
        <p class="aw-flag">Unverified default: no release manifest pins these bytes.</p>
      )}
      {earlier && (
        <p class="aw-flag">
          Showing an earlier published version of this map, pinned by the unit you opened.
        </p>
      )}
      <details class="aw-about">
        <summary>About this map</summary>
        <dl class="aw-dl">
          <dt>Model</dt>
          <dd>{modelLine(meta)}</dd>
          <dt>Unit</dt>
          <dd class="aw-mono">{str(meta.unit) ?? "Not recorded"}</dd>
          {str(meta.geometry) && (
            <>
              <dt>Geometry</dt>
              <dd>{str(meta.geometry)}</dd>
            </>
          )}
          {str(meta.sae_release) && (
            <>
              <dt>SAE release</dt>
              <dd class="aw-mono">{str(meta.sae_release)}</dd>
            </>
          )}
          {str(meta.curation) && (
            <>
              <dt>Curation</dt>
              <dd>
                {cur
                  ? `The first ${fmtInt(cur.kept)} of ${fmtInt(cur.total)} — a partial subset, not a representative sample`
                  : str(meta.curation)}
              </dd>
            </>
          )}
          <dt>Clusters</dt>
          <dd>
            {fmtInt(ds.columns.clusters.length)}
            {noise !== null && (
              <>
                {" "}· {pct(noise)} unclustered <span class="aw-raw">(noise_fraction {noise})</span>
              </>
            )}
          </dd>
          <dt>Cluster names</dt>
          <dd>{str(meta.namer) ?? "Not recorded"}</dd>
          <dt>Created</dt>
          <dd>{str(meta.created) ?? "Not recorded"}</dd>
          <dt>Artifact</dt>
          <dd class="aw-mono" title={ds.sha256 ?? undefined}>
            {ds.sha256 ? `sha256 ${shortSha(ds.sha256)}…` : `not hashed (${ds.hashError ?? "unknown"})`}
          </dd>
        </dl>
      </details>
    </header>
  );
}

/* ── results table ───────────────────────────────────────────────────── */

/** A Layer column earns its width only when rows differ; a single-layer map
 *  states its layer once, in the header. Cached per dataset. */
const layerCache = new WeakMap<Dataset, boolean>();
function layerVaries(ds: Dataset): boolean {
  let v = layerCache.get(ds);
  if (v === undefined) {
    const l = ds.columns.source.layer;
    v = false;
    for (let i = 1; i < l.length; i++) {
      if (!Object.is(l[i], l[0])) {
        v = true;
        break;
      }
    }
    layerCache.set(ds, v);
  }
  return v;
}

export function ResultsTable({ ds, heading = "Units" }: { ds: Dataset; heading?: string }) {
  const { text, results } = $mapQuery.value;
  const meta = metaOf(ds);
  const n = ds.columns.count;
  const total = results ? results.total : n;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const page = Math.min($page.value, pages - 1);
  const start = page * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, total);
  const rows: number[] = [];
  for (let i = start; i < end; i++) rows.push(results ? results.matchIds[i]! : i);
  const sel = $selection.value;
  const selRow = sel?.kind === "point" ? sel.id : -1;
  const hasLayer = layerVaries(ds);
  const setQuery = (t: string) => appStore.getState().setMapQuery(t);
  const unlabelled = isUnlabelled(meta);

  return (
    <section class="rt" aria-labelledby="rt-heading">
      <h3 id="rt-heading" class="aw-subtitle">
        {heading}
      </h3>
      <label class="rt-search-label" for="rt-search">
        Search labels
      </label>
      <div class="rt-search-row">
        <input
          id="rt-search"
          class="rt-search"
          type="search"
          autocomplete="off"
          spellcheck={false}
          placeholder={unlabelled ? "This map has no labels to search" : "e.g. numbers"}
          value={text}
          onInput={(e) => setQuery((e.currentTarget as HTMLInputElement).value)}
          onKeyDown={(e) => {
            if (e.key === "Escape" && text) {
              e.stopPropagation();
              setQuery("");
            }
          }}
        />
      </div>
      <p class="rt-summary" role="status" aria-live="polite">
        {results
          ? results.total === 0
            ? `No label contains “${text.trim()}”.`
            : `${fmtInt(results.total)} of ${fmtInt(n)} labels contain “${text.trim()}”. Showing ${fmtInt(start + 1)}–${fmtInt(end)}.`
          : `Showing ${fmtInt(start + 1)}–${fmtInt(end)} of ${fmtInt(n)}, in artifact order.`}
      </p>
      {results && results.total === 0 ? (
        <div class="rt-empty">
          <p>Search matches label text exactly (substring, any case). It does not search meaning.</p>
          <button type="button" class="aw-btn" onClick={() => setQuery("")}>
            Clear search
          </button>
        </div>
      ) : (
        <div class="rt-scroll">
          <table class="rt-table">
            <caption class="sr-only">
              {results ? `Units whose label contains “${text.trim()}”` : "All units"}, page {page + 1} of {pages}
            </caption>
            <thead>
              <tr>
                <th scope="col">{unlabelled ? "Unit" : "Label"}</th>
                <th scope="col" class="rt-num">
                  Index
                </th>
                {hasLayer && (
                  <th scope="col" class="rt-num">
                    Layer
                  </th>
                )}
                <th scope="col">Cluster</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const u = sourceUnit(ds.columns, r)!;
                const title = unitTitle(meta, u);
                const selected = r === selRow;
                return (
                  <tr key={r} class={selected ? "is-selected" : undefined} aria-current={selected ? "true" : undefined}>
                    <th scope="row">
                      <button
                        type="button"
                        class="rt-inspect"
                        data-row={r}
                        aria-label={`Inspect ${title}`}
                        onClick={(e) => inspectRow(r, e.currentTarget as HTMLElement)}
                      >
                        {title}
                      </button>
                    </th>
                    <td class="rt-num">{u.unitIndex ?? "—"}</td>
                    {hasLayer && <td class="rt-num">{u.layer ?? "—"}</td>}
                    <td class={u.clusterId < 0 ? "rt-cluster is-noise" : "rt-cluster"}>
                      {clusterTitle(ds, u.clusterId)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {pages > 1 && !(results && results.total === 0) && (
        <nav class="rt-pager" aria-label="Results pages">
          <button type="button" class="aw-btn" disabled={page === 0} onClick={() => ($page.value = page - 1)}>
            Previous
          </button>
          <span class="rt-page">
            Page {page + 1} of {fmtInt(pages)}
          </span>
          <button
            type="button"
            class="aw-btn"
            disabled={page >= pages - 1}
            onClick={() => ($page.value = page + 1)}
          >
            Next
          </button>
        </nav>
      )}
    </section>
  );
}

/* ── toolbar and disclosures ─────────────────────────────────────────── */

function datasetOptionLabel(id: string): string {
  const art = currentArtifact(activeManifest(), id);
  return art?.label ? `${art.label}` : id;
}

function Toolbar() {
  const gpu = $renderer.value !== "unavailable";
  return (
    <div class="aw-toolbar">
      <SelectRow
        label="Map"
        value={$datasetId.value ?? $pendingDatasetId.value ?? ""}
        options={[
          ...($datasetId.value || $pendingDatasetId.value ? [] : [{ value: "", label: "Choose a map…" }]),
          ...$datasets.value.map((d) => ({ value: d.id, label: datasetOptionLabel(d.id) })),
        ]}
        onChange={(id) => id && requestDataset(id)}
      />
      <div class="aw-dims" role="group" aria-label="Map dimensions">
        {([2, 3] as const).map((d) => (
          <button
            key={d}
            type="button"
            class={$dims.value === d ? "aw-seg is-on" : "aw-seg"}
            aria-pressed={$dims.value === d}
            disabled={!gpu}
            onClick={() => appStore.getState().setDims(d)}
          >
            {d === 2 ? "2D map" : "3D"}
          </button>
        ))}
        <a class="aw-advanced" href={advancedHref($datasetId.value)}>
          Advanced analysis<span aria-hidden="true"> ↗</span>
          <span class="sr-only"> (opens Research)</span>
        </a>
      </div>
    </div>
  );
}

const TOGGLE_ROWS: { key: keyof Toggles; label: string }[] = [
  { key: "territories", label: "Territories" },
  { key: "labels", label: "Cluster names" },
  { key: "beams", label: "Connections" },
  { key: "halos", label: "Halos" },
  { key: "noise", label: "Unclustered points" },
  { key: "legend", label: "Legend" },
];

function DisplayOptions() {
  const toggles = $toggles.value;
  const settings = $settings.value;
  const handReason = handControlUnavailableReason();
  const st = appStore.getState();
  return (
    <details class="aw-disclosure">
      <summary>Display</summary>
      <div class="aw-disclosure-body">
        {TOGGLE_ROWS.map((r) => (
          <ToggleRow key={r.key} label={r.label} checked={toggles[r.key]} onChange={(v) => st.setToggle(r.key, v)} />
        ))}
        <SliderRow
          label="Point scale"
          value={settings.pointScale}
          min={0.5}
          max={2}
          step={0.05}
          format={(v) => `${v.toFixed(2)}×`}
          onChange={(v) => st.setSetting("pointScale", v)}
        />
        <SliderRow
          label="Membership floor"
          value={settings.confidenceFloor}
          min={0}
          max={1}
          step={0.01}
          format={(v) => `${Math.round(v * 100)}%`}
          onChange={(v) => st.setSetting("confidenceFloor", v)}
        />
        <ToggleRow
          label="Hand control"
          checked={settings.handTracking}
          disabled={handReason !== null}
          hint={handReason ?? "webcam gestures — video never leaves this machine"}
          onChange={(v) => st.setSetting("handTracking", v)}
        />
        {/* Effects ride the hand that is NOT steering, so they are opt-in and
            only offered once the rig is actually on. */}
        {settings.handTracking && (
          <ToggleRow
            label="Hand effects"
            checked={settings.handEffects}
            hint="your free hand can throw a shockwave or snap the cloud bright"
            onChange={(v) => st.setSetting("handEffects", v)}
          />
        )}
      </div>
    </details>
  );
}

function Lenses() {
  return (
    <details class="aw-disclosure">
      <summary>Lenses and directions</summary>
      <div class="aw-disclosure-body">
        <ChannelLens />
        <DirectionMaker />
      </div>
    </details>
  );
}

/* ── import a saved finding ──────────────────────────────────────────── */

export async function importFindingFile(file: File): Promise<void> {
  const st = appStore.getState();
  const reject = (code: string, message: string) =>
    st.setPin({ status: "error", pin: null, source: "import", code, title: "This file can't be opened", message });
  // the size is checked before a single byte is read
  if (file.size > FINDING_MAX_BYTES) {
    const r = parseFindingText("", file.size);
    if (!r.ok) reject(r.code, r.message);
    return;
  }
  let text: string;
  try {
    text = await file.text();
  } catch (e) {
    reject("unreadable", `The file could not be read (${e instanceof Error ? e.message : e}).`);
    return;
  }
  const r = parseFindingText(text, file.size);
  if (!r.ok) {
    reject(r.code, r.message);
    return;
  }
  requestOpenPinned(findingPin(r.finding), "import", r.finding);
}

function FindingImport() {
  const input = useRef<HTMLInputElement>(null);
  return (
    <div class="aw-import">
      <button type="button" class="aw-btn" onClick={() => input.current?.click()}>
        Open saved finding…
      </button>
      <input
        ref={input}
        class="sr-only"
        type="file"
        accept=".json,application/json"
        tabIndex={-1}
        aria-hidden="true"
        data-testid="finding-file"
        onChange={(e) => {
          const el = e.currentTarget as HTMLInputElement;
          const f = el.files?.[0];
          el.value = "";
          if (f) void importFindingFile(f);
        }}
      />
      <button type="button" class="aw-btn aw-btn-quiet" onClick={() => appStore.getState().setSettingsOpen(true)}>
        All settings…
      </button>
    </div>
  );
}

/* ── pin notice ──────────────────────────────────────────────────────── */

export function PinNotice() {
  const pin = $pin.value;
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (pin.status === "error") ref.current?.focus();
  }, [pin]);
  if (pin.status === "pending") {
    return (
      <div class="pin-notice is-pending" role="status">
        Verifying the saved unit against its published artifact…
      </div>
    );
  }
  if (pin.status !== "error") return null;
  const st = appStore.getState();
  const known = pin.pin && $datasets.value.some((d) => d.id === pin.pin!.datasetId);
  return (
    <div class="pin-notice" role="alert" tabIndex={-1} ref={ref} aria-labelledby="pin-notice-title">
      <p id="pin-notice-title" class="pin-notice-title">
        {pin.title}
      </p>
      <p class="pin-notice-body">{pin.message}</p>
      {pin.pin && (
        <dl class="pin-notice-dl">
          <dt>Requested</dt>
          <dd class="aw-mono">
            {pin.pin.datasetId} · point {pin.pin.pointId} · {pin.pin.unitKind} #{pin.pin.unitIndex}
          </dd>
          {pin.expected && (
            <>
              <dt>Expected artifact</dt>
              <dd class="aw-mono" title={pin.expected}>
                sha256 {shortSha(pin.expected)}…
              </dd>
            </>
          )}
          {pin.actual && (
            <>
              <dt>Published here</dt>
              <dd class="aw-mono" title={pin.actual}>
                sha256 {shortSha(pin.actual)}…
              </dd>
            </>
          )}
        </dl>
      )}
      {pin.conflicts && pin.conflicts.length > 0 && (
        <p class="pin-notice-body">Fields that differ: {pin.conflicts.join(", ")}.</p>
      )}
      <div class="pin-notice-actions">
        {known && pin.code !== "conflict" && (
          <button
            type="button"
            class="aw-btn"
            onClick={() => {
              st.setPin({ status: "none" });
              requestDataset(pin.pin!.datasetId);
            }}
          >
            Open the latest map
          </button>
        )}
        {!$dataset.value && !known && (
          <button
            type="button"
            class="aw-btn"
            onClick={() => {
              st.setPin({ status: "none" });
              requestStarter();
            }}
          >
            Open the starter map
          </button>
        )}
        <button
          type="button"
          class="aw-btn aw-btn-quiet"
          onClick={() => {
            st.setPin({ status: "none" });
            document.getElementById("sel-map")?.focus();
          }}
        >
          Choose a map
        </button>
      </div>
    </div>
  );
}

/* ── the panel ───────────────────────────────────────────────────────── */

function LoadState() {
  const loading = $loading.value;
  const pending = $pendingDatasetId.value;
  const err = $loadError.value;
  if (err) {
    return (
      <div class="aw-state" role="alert">
        <p>{err.message}</p>
        {err.expected && (
          <p class="aw-mono">
            expected sha256 {shortSha(err.expected)}… · got {shortSha(err.actual)}…
          </p>
        )}
        {err.kind !== "no-starter" && err.kind !== "unknown-model" && (
          <button type="button" class="aw-btn" onClick={() => requestRetryLoad()}>
            Retry
          </button>
        )}
      </div>
    );
  }
  if (pending) {
    const pctDone = loading.total > 0 ? ` — ${Math.round((loading.loaded / loading.total) * 100)}%` : "";
    return (
      <p class="aw-state" role="status">
        Opening {datasetOptionLabel(pending)}
        {pctDone}…
      </p>
    );
  }
  return null;
}

export function AtlasWorkspace({ hidden = false }: { hidden?: boolean }) {
  const ds = $dataset.value;
  const id = $datasetId.value;
  return (
    <aside class={ds ? "atlas-workspace has-data" : "atlas-workspace"} aria-label="Atlas workspace" hidden={hidden}>
      {ds && id ? <EvidenceHeader ds={ds} datasetId={id} /> : <h2 class="aw-title">No map open</h2>}
      <Toolbar />
      <LoadState />
      {$renderer.value === "unavailable" && (
        <p class="aw-note">
          The map needs WebGL or WebGPU, which this browser does not provide. Every unit is still
          in the list below, with the same evidence.
        </p>
      )}
      {ds && <ResultsTable ds={ds} />}
      {ds && <DisplayOptions />}
      {ds && <Lenses />}
      <FindingImport />
    </aside>
  );
}
