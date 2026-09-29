/** Research's entry to Internals: an explicit model-export chooser, then the
 *  first analysis task beside its chart. Both read only what the server
 *  actually publishes — the dataset index's `has_interp` flag, each export's
 *  small `interp/index.json`, and the release manifest's pinned bundle — so
 *  nothing here promises an analysis the export cannot show, and opening
 *  Research never downloads a model map. */

import { signal } from "@preact/signals";
import { useEffect } from "preact/hooks";
import { appStore } from "../../app/store";
import { activeManifest, manifestStatus } from "../../data/experience";
import { interpBase, loadInterpIndex, primeBundle, type InterpIndex } from "../../data/interp";
import { INTERP_FEATURES } from "../../scene/interp/registry";
import { availability, availableCount } from "../../scene/interp/requirements";
import { $datasets } from "../state";
import {
  CAVEAT_OPTIONS,
  QUANTITY_OPTIONS,
  TASK_STEPS,
  downloadBytes,
  formatPeriod,
  formatPower,
  loadIntroProgress,
  pendingSteps,
  saveIntroProgress,
  stepDone,
  topFrequencies,
  verifyBundle,
  type BundleCheck,
  type IntroProgress,
  type Option,
  type ResearchIntro,
} from "./intro";

/* ── shared state ─────────────────────────────────────────────────────── */

/** Each export's index, by model id. undefined = not asked yet or in flight,
 *  null = the model has no internals export. */
export const $exportIndex = signal<Record<string, InterpIndex | null>>({});
const asked = new Set<string>();

export function ensureExportIndex(model: string): void {
  if (asked.has(model)) return;
  asked.add(model);
  loadInterpIndex(model)
    .then((idx) => {
      $exportIndex.value = { ...$exportIndex.value, [model]: idx };
    })
    .catch(() => {
      $exportIndex.value = { ...$exportIndex.value, [model]: null };
    });
}

/** The pinned bundle's verification, once per page (Try again re-runs it). */
export const $introCheck = signal<BundleCheck>({ state: "pending" });
let checking: string | null = null;

export function researchIntro(): ResearchIntro | null {
  return activeManifest()?.research_intro ?? null;
}

export function ensureIntroCheck(ri: ResearchIntro, force = false): void {
  if (!force && (checking === ri.sha256 || $introCheck.value.state !== "pending")) return;
  checking = ri.sha256;
  $introCheck.value = { state: "pending" };
  void verifyBundle(ri).then((r) => {
    if (checking !== ri.sha256) return;
    // Hand the chart exactly the verified bytes: seed the bundle cache under
    // the URL its loader asks for, but only when that URL IS the pinned file.
    if (r.state === "verified" && r.url === `${interpBase(ri.dataset_id)}/fourier.json`)
      primeBundle(r.url, r.json);
    $introCheck.value = r;
  });
}

/** Is the pinned-task view on screen (so the chart must wait for the check)? */
export function isIntroView(model: string | null, featureId: string): ResearchIntro | null {
  const ri = researchIntro();
  return ri && model === ri.dataset_id && featureId === ri.feature ? ri : null;
}

const FEATURE_IDS = INTERP_FEATURES.map((f) => f.id);

/** Open an export: the recommended one opens on its task; any other opens on
 *  the pinned analysis when it has it, else on its first available one. */
/** An analysis asked for before any export was chosen — from a Methods card
 *  or a link that names a feature but no model. The chooser marks which
 *  exports include it, and choosing one opens it rather than the first task. */
export const $requestedFeature = signal<string | null>(null);

export function requestFeature(id: string | null): void {
  $requestedFeature.value = id !== null && (FEATURE_IDS as readonly string[]).includes(id) ? id : null;
}

export function openExport(model: string): void {
  const st = appStore.getState();
  const ri = researchIntro();
  const idx = $exportIndex.value[model];
  const usable = (id: string | null | undefined): id is string =>
    !!id && ["available", "live"].includes(availability(id, idx).state);
  const requested = $requestedFeature.value;
  const preferred = ri?.feature ?? "fourier-atlas";
  const first = usable(requested)
    ? requested
    : usable(preferred)
      ? preferred
      : (FEATURE_IDS.find((id) => availability(id, idx).state === "available") ?? preferred);
  $requestedFeature.value = null;
  st.setInterpModel(model);
  st.setInterpFeature(first);
}

const featureLabel = (id: string): string =>
  INTERP_FEATURES.find((f) => f.id === id)?.label ?? id;

/* ── the chooser ──────────────────────────────────────────────────────── */

export function ResearchChooser() {
  const datasets = $datasets.value;
  const indexes = $exportIndex.value;
  const ri = researchIntro();
  const withExport = datasets.filter((d) => d.has_interp === true);
  const mapOnly = datasets.filter((d) => d.has_interp !== true);
  // recommended first, then the index's own order
  withExport.sort((a, b) => Number(b.id === ri?.dataset_id) - Number(a.id === ri?.dataset_id));
  useEffect(() => {
    for (const d of withExport) ensureExportIndex(d.id);
  }, [withExport.map((d) => d.id).join("|")]);

  const loading = datasets.length === 0 && manifestStatus().state === "unloaded";
  const requested = $requestedFeature.value;

  return (
    <div class="research-chooser" role="main" aria-labelledby="research-chooser-title">
      <h1 id="research-chooser-title" class="research-chooser-title">
        Choose a model export
      </h1>
      <p class="research-chooser-lede">
        Each Internals analysis reads files exported from one model. Choose an export to see
        which of the {FEATURE_IDS.length} registered analyses it can show. No model map is
        downloaded here.
      </p>
      {requested && (
        <p class="research-chooser-request" role="status">
          You asked for <strong>{featureLabel(requested)}</strong>. Each export below says
          whether it includes that analysis; choosing one opens it there.{" "}
          <button type="button" class="research-link" onClick={() => requestFeature(null)}>
            Start with the first task instead
          </button>
        </p>
      )}
      {loading ? (
        <p class="research-chooser-status" role="status">
          Reading the list of published exports…
        </p>
      ) : withExport.length === 0 ? (
        <p class="research-chooser-status" role="status">
          This server publishes no internals export. Maps can still be compared under
          Comparisons.
        </p>
      ) : (
        <ul class="research-chooser-list">
          {withExport.map((d) => {
            const idx = indexes[d.id];
            const recommended = d.id === ri?.dataset_id;
            const n = idx === undefined ? null : availableCount(FEATURE_IDS, idx);
            const has =
              requested === null || idx === undefined
                ? null
                : ["available", "live"].includes(availability(requested, idx).state);
            return (
              <li key={d.id}>
                <button
                  type="button"
                  class={`research-export${recommended ? " is-recommended" : ""}`}
                  onClick={() => openExport(d.id)}
                  aria-describedby={`export-${d.id}-count${has !== null ? ` export-${d.id}-has` : ""}${recommended && requested === null ? ` export-${d.id}-why` : ""}`}
                >
                  <span class="research-export-head">
                    <span class="research-export-name">{d.id}</span>
                    {recommended && <span class="research-export-badge">Recommended</span>}
                  </span>
                  <span class="research-export-count" id={`export-${d.id}-count`}>
                    {idx === null
                      ? "Export index could not be read"
                      : n === null
                        ? "Checking which analyses it has…"
                        : `${n} of ${FEATURE_IDS.length} analyses available`}
                  </span>
                  {has !== null && (
                    <span
                      class={`research-export-has${has ? "" : " is-missing"}`}
                      id={`export-${d.id}-has`}
                    >
                      {has
                        ? `Includes ${featureLabel(requested!)}`
                        : `Does not include ${featureLabel(requested!)}`}
                    </span>
                  )}
                  {recommended && requested === null && (
                    <span class="research-export-why" id={`export-${d.id}-why`}>
                      Opens on a first task: Position Patterns by Frequency, read from a{" "}
                      {(ri!.bytes / 1000).toFixed(1)} kB file checked against the release.
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {mapOnly.length > 0 && (
        <details class="research-chooser-more">
          <summary>
            {mapOnly.length} model{mapOnly.length === 1 ? " has" : "s have"} a map but no
            internals export
          </summary>
          <p>
            {mapOnly.map((d) => d.id).join(", ")}. Their maps open under{" "}
            <button
              type="button"
              class="research-link"
              onClick={() => appStore.getState().setPage("map")}
            >
              Comparisons
            </button>
            .
          </p>
        </details>
      )}
    </div>
  );
}

/* ── the first task ───────────────────────────────────────────────────── */

const progress = signal<IntroProgress | null>(null);

function update(p: Partial<IntroProgress>): void {
  if (!progress.value) return;
  progress.value = { ...progress.value, ...p };
  saveIntroProgress(progress.value);
}

function Choice(props: {
  name: string;
  legend: string;
  options: readonly Option[];
  value: string | null;
  onPick: (id: string) => void;
}) {
  const picked = props.options.find((o) => o.id === props.value);
  return (
    <fieldset class="research-q">
      <legend>{props.legend}</legend>
      {props.options.map((o) => (
        <label key={o.id} class={`research-opt${props.value === o.id ? " is-picked" : ""}`}>
          <input
            type="radio"
            name={props.name}
            value={o.id}
            checked={props.value === o.id}
            onChange={() => props.onPick(o.id)}
          />
          <span>{o.label}</span>
        </label>
      ))}
      <p
        class={`research-feedback${picked ? (picked.correct ? " is-right" : " is-wrong") : ""}`}
        role="status"
      >
        {picked ? picked.why : ""}
      </p>
    </fieldset>
  );
}

const STEP_LABEL: Record<(typeof TASK_STEPS)[number], string> = {
  quantity: "Name the measured quantity",
  value: "Record one value",
  caveat: "Name what it cannot show",
  export: "Save the exact file",
};

export function ResearchTask(props: { ri: ResearchIntro; staticTier: boolean }) {
  const { ri } = props;
  const check = $introCheck.value;
  useEffect(() => {
    ensureIntroCheck(ri);
  }, [ri.sha256]);
  if (!progress.value || progress.value.sha256 !== ri.sha256) progress.value = loadIntroProgress(ri.sha256);
  const p = progress.value!;
  const shortSha = `${ri.sha256.slice(0, 8)}…${ri.sha256.slice(-4)}`;

  if (check.state === "pending")
    return (
      <aside class="research-task" aria-labelledby="research-task-title">
        <h2 id="research-task-title" class="research-task-title">First task</h2>
        <p class="research-task-status" role="status">
          Checking {ri.bundle_path} against the release manifest…
        </p>
      </aside>
    );

  if (check.state !== "verified") {
    const missing = check.state === "missing";
    return (
      <aside class="research-task is-problem" aria-labelledby="research-task-title">
        <h2 id="research-task-title" class="research-task-title">
          {missing ? "The analysis file is not on this server" : "The file here is not the released one"}
        </h2>
        <div role="alert">
          {missing ? (
            <p>
              Research expected <code>{ri.bundle_path}</code> ({ri.bytes.toLocaleString("en-US")} bytes),
              but {check.status === null ? "the request failed" : `the server answered ${check.status}`}.
              Nothing was substituted, so no chart is shown.
            </p>
          ) : (
            <p>
              <code>{ri.bundle_path}</code> is {check.bytes.toLocaleString("en-US")} bytes with SHA-256{" "}
              <code>{check.actual.slice(0, 12)}…</code>. The release pins{" "}
              {ri.bytes.toLocaleString("en-US")} bytes with <code>{ri.sha256.slice(0, 12)}…</code>. The
              chart is withheld rather than drawn from different numbers.
            </p>
          )}
        </div>
        <div class="research-actions">
          <button type="button" class="research-btn is-primary" onClick={() => ensureIntroCheck(ri, true)}>
            Try again
          </button>
          <button
            type="button"
            class="research-btn"
            onClick={() => appStore.getState().setInterpModel(null)}
          >
            Choose another model
          </button>
        </div>
      </aside>
    );
  }

  const rows = topFrequencies(check.json, 8);
  const done = TASK_STEPS.filter((s) => stepDone(s, p, rows));
  const left = pendingSteps(p, rows);
  const recorded = rows.find((r) => r.index === p.value);
  const d = check.json.meta.d;

  return (
    <aside class="research-task" aria-labelledby="research-task-title">
      <h2 id="research-task-title" class="research-task-title">
        First task: read one measurement and its limit
      </h2>
      <p class="research-task-source">
        Verified <code>{ri.bundle_path}</code> · {ri.bytes.toLocaleString("en-US")} bytes · SHA-256{" "}
        <code title={ri.sha256}>{shortSha}</code>
      </p>
      <ol class="research-steps" aria-label="Task progress">
        {TASK_STEPS.map((s) => (
          <li key={s} class={done.includes(s) ? "is-done" : ""}>
            <span aria-hidden="true">{done.includes(s) ? "✓" : "○"}</span> {STEP_LABEL[s]}
            <span class="sr-only">{done.includes(s) ? ", done" : ", not done"}</span>
          </li>
        ))}
      </ol>

      <Choice
        name="research-quantity"
        legend="1. What does this chart measure?"
        options={QUANTITY_OPTIONS}
        value={p.quantity}
        onPick={(id) => update({ quantity: id })}
      />

      <fieldset class="research-q">
        <legend>2. Record one frequency and its power</legend>
        <p class="research-hint">
          The {rows.length} strongest frequencies in the file. Frequency counts cycles per{" "}
          {check.json.meta.n_ctx.toLocaleString("en-US")}-position window, period is positions per cycle,
          and the last column counts embedding dimensions whose strongest pattern is there.{" "}
          {props.staticTier
            ? "The chart needs WebGL or WebGPU, so read the values here."
            : "Hovering the gold curve at the same angle shows the same values."}
        </p>
        <div class="research-table-wrap" tabIndex={0} role="group" aria-label="Strongest frequencies, scrollable">
          <table class="research-table">
            <caption class="sr-only">
              Strongest frequencies in {ri.bundle_path}. Frequency 0, the removed mean, is left out.
            </caption>
            <thead>
              <tr>
                <th scope="col">Record</th>
                <th scope="col">Frequency</th>
                <th scope="col">Period</th>
                <th scope="col">Mean power</th>
                <th scope="col">Dims peaking</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.index} class={p.value === r.index ? "is-picked" : ""}>
                  <td>
                    <input
                      type="radio"
                      name="research-value"
                      checked={p.value === r.index}
                      aria-label={`Record frequency ${r.freq}`}
                      onChange={() => update({ value: r.index })}
                    />
                  </td>
                  <th scope="row">{r.freq}</th>
                  <td>{formatPeriod(r.period)}</td>
                  <td>{formatPower(r.power)}</td>
                  <td>
                    {r.dims} of {d}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p class="research-feedback is-right" role="status">
          {recorded
            ? `Recorded: frequency ${recorded.freq} repeats every ${formatPeriod(recorded.period)} positions, with mean power ${formatPower(recorded.power)}. ${recorded.dims} of ${d} dimensions have their strongest pattern here.`
            : ""}
        </p>
      </fieldset>

      <Choice
        name="research-caveat"
        legend="3. Which of these can the chart NOT show?"
        options={CAVEAT_OPTIONS}
        value={p.caveat}
        onPick={(id) => update({ caveat: id })}
      />

      <fieldset class="research-q">
        <legend>4. Save the exact file</legend>
        <p class="research-hint">
          The download is the verified bytes as published, unchanged, so a result can be traced
          to this release.
        </p>
        <button
          type="button"
          class="research-btn is-primary"
          onClick={() => {
            downloadBytes(check.bytes, `${ri.dataset_id}-fourier.json`);
            update({ exported: true });
          }}
        >
          Download {ri.dataset_id}-fourier.json
        </button>
        <p class="research-feedback is-right" role="status">
          {p.exported ? `Saved. Its SHA-256 is ${ri.sha256}.` : ""}
        </p>
      </fieldset>

      <div class="research-complete" role="status">
        {left.length === 0 && (
          <>
            <h3>Task complete</h3>
            <p>
              You named the quantity, recorded frequency {recorded?.freq} with mean power{" "}
              {recorded ? formatPower(recorded.power) : ""}, named what the chart cannot show and
              saved the exact file.
            </p>
            <div class="research-actions">
              <button
                type="button"
                class="research-btn"
                onClick={() =>
                  (document.querySelector(".interp-feature:not(.is-active)") as HTMLElement | null)?.focus()
                }
              >
                Browse the other analyses
              </button>
              <button
                type="button"
                class="research-btn"
                onClick={() => appStore.getState().setPage("guide")}
              >
                Read the methods
              </button>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
