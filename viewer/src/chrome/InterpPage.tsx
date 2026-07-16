/** Internals — the mechanistic-interpretability gallery. A dedicated page with
 *  its own canvas, RAF loop, and feature rail. One InterpDriver owns the canvas
 *  at a time; each renders exactly ONE real computed quantity from an interp
 *  bundle (out/<model>/interp/*.json) and exposes exact hover values. The rail
 *  only lists features whose driver is live and backed by real data — an
 *  unimplemented feature never appears, so the page can't overstate what it shows.
 *
 *  The active model is the current dataset id (bundles are per-model). If a model
 *  has no interp export, the driver's setModel rejects and we say so plainly. */

import { signal, useSignal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";
import { requestDataset } from "../app/actions";
import { appStore } from "../app/store";
import {
  LIVE_TRACE_PREFIX,
  cachedBundle,
  hasInterp,
  isLiveTrace,
  loadInterpIndex,
  putLiveTrace,
  registerLivePrompt,
  startBundleCapture,
  takeBundleCapture,
  type TraceBundle,
} from "../data/interp";
import type { DatasetEntry } from "../data/schema";
import type { InterpDriver, InterpGroup } from "../scene/interp/InterpDriver";
import { GROUP_LABEL, INTERP_FEATURES, findFeature } from "../scene/interp/registry";
import { SelectRow } from "./controls";
import {
  $capabilities,
  $datasetId,
  $datasets,
  $interp,
  $interpSelection,
  $loading,
  $probing,
  $tour,
  $viewMode,
} from "./state";
import { TOURS, applyTourStep, findTour } from "./tours";

interface TraceEntry {
  slug: string;
  prompt: string;
}

/** 2b — traces computed by the local live server on typed prompts, per model.
 *  Module-level so switching features (which remounts the page's effects)
 *  keeps the session's live prompts in the trace bar. The payloads live in
 *  the bundle cache (putLiveTrace); this is just the chip list. */
const $liveTraces = signal<Record<string, TraceEntry[]>>({});

/** Which datasets have an interp export (out/<id>/interp/index.json)? The
 *  discovery index carries a `has_interp` flag, so the model picker can say up
 *  front which models the Internals views will work for with zero extra
 *  requests. Only entries from an older index that predate the flag fall back
 *  to a per-model network probe. null = not resolved yet. */
const $interpAvail = signal<Record<string, boolean> | null>(null);
let availProbe: string | null = null;
function probeInterpAvail(entries: DatasetEntry[]): void {
  const key = entries.map((e) => e.id).join("|");
  if (availProbe === key) return;
  availProbe = key;
  const known: Record<string, boolean> = {};
  const unknown: DatasetEntry[] = [];
  for (const e of entries) {
    if (typeof e.has_interp === "boolean") known[e.id] = e.has_interp;
    else unknown.push(e);
  }
  if (unknown.length === 0) {
    $interpAvail.value = known; // fully answered by the index — no network
    return;
  }
  Promise.all(unknown.map(async (e) => [e.id, await hasInterp(e.id)] as const)).then(
    (pairs) => {
      $interpAvail.value = { ...known, ...Object.fromEntries(pairs) };
    },
  );
}

export function InterpPage() {
  const interp = $interp.value;
  const model = $datasetId.value;
  const caps = $capabilities.value;
  const tier = caps?.tier ?? "webgl";
  const feature = findFeature(interp.featureId);

  const status = useSignal<"loading" | "ready" | "error">("loading");
  // bundle URLs the active view was computed from (captured during the
  // driver's load) — powers the legend's "download data" affordance
  const dataUrls = useSignal<string[]>([]);
  // Which real step the loading is in: spinning up the renderer vs fetching
  // the bundle + computing the layout. Two awaits, two honest stages.
  const phase = useSignal<"renderer" | "data">("renderer");
  const errMsg = useSignal("");
  const traces = useSignal<TraceEntry[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const driverRef = useRef<InterpDriver | null>(null);

  // Legend disclosure: null = auto (open on wide stages, collapsed at the
  // narrow breakpoint where the card would occlude the plot). A user toggle
  // overrides auto and stays sticky for the session — transient view chrome,
  // not a persisted setting.
  const narrow = useSignal(false);
  const legendOpen = useSignal<boolean | null>(null);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 900px)");
    narrow.value = mq.matches;
    const onChange = (e: MediaQueryListEvent) => {
      narrow.value = e.matches;
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  const legendIsOpen =
    legendOpen.value ?? (!narrow.value && feature?.legendCollapsed !== true);

  // Per-trace features render one bundled prompt at a time (all of the forward
  // group, plus anything flagged perTrace). Load the trace list for the model
  // so the selector can offer them; resolve "" to the first slug.
  const isForward =
    (feature?.group === "forward" || feature?.perTrace === true) && feature?.ownPrompts !== true;
  const avail = $interpAvail.value;
  useEffect(() => {
    if ($datasets.value.length) probeInterpAvail($datasets.value);
  }, [$datasets.value]);
  // hasIdx: did out/<model>/interp/index.json load? Distinguishes "this model
  // has no internals export at all" from "the export exists but this feature's
  // bundle isn't in it" (SAE/trained bundles are legitimately gpt2-only — the
  // res-jb SAE release covers gpt2), so the error card can tell the truth.
  const hasIdx = useSignal<boolean | null>(null);
  useEffect(() => {
    if (!model) return;
    let ok = true;
    hasIdx.value = null;
    loadInterpIndex(model)
      .then((idx) => {
        if (!ok) return;
        traces.value = idx.traces ?? [];
        hasIdx.value = true;
      })
      .catch(() => {
        if (!ok) return;
        traces.value = [];
        hasIdx.value = false;
      });
    return () => {
      ok = false;
    };
  }, [model]);
  const resolvedTrace = isForward
    ? interp.traceSlug || traces.value[0]?.slug || ""
    : "";

  // driver lifecycle: (re)build whenever feature, model, or GPU tier changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    // Size from the STAGE container, never the canvas: deck.gl resizes the
    // canvas to its own width/height props, so measuring the canvas would feed
    // deck's size back into itself (a shrinking loop). The stage is the truth.
    const host = canvas?.parentElement;
    if (!canvas || !overlay || !host || !model || !feature) return;

    let disposed = false;
    let raf = 0;
    let last = performance.now();
    const t0 = last;
    status.value = "loading";
    phase.value = "renderer";
    errMsg.value = "";
    dataUrls.value = [];

    const dprOf = () => Math.min(window.devicePixelRatio || 1, 2);
    const sizeNow = () => {
      const r = host.getBoundingClientRect();
      return { w: Math.max(1, Math.round(r.width)), h: Math.max(1, Math.round(r.height)) };
    };

    (async () => {
      const d = feature.create();
      await d.init(canvas, tier, overlay);
      if (disposed) {
        d.dispose();
        return;
      }
      const sz = sizeNow();
      d.resize(sz.w, sz.h, dprOf());
      driverRef.current = d;
      phase.value = "data";
      // record which bundle files this load actually reads, so the legend's
      // download button can offer exactly the view's source data (1d)
      const cap = startBundleCapture();
      try {
        await d.setModel(model, resolvedTrace);
      } catch (e) {
        takeBundleCapture(cap);
        if (disposed) {
          driverRef.current?.dispose();
          driverRef.current = null;
          return;
        }
        status.value = "error";
        errMsg.value = e instanceof Error ? e.message : String(e);
        return;
      }
      if (!disposed) dataUrls.value = takeBundleCapture(cap);
      else takeBundleCapture(cap);
      if (disposed) {
        driverRef.current?.dispose();
        driverRef.current = null;
        return;
      }
      status.value = "ready";
      // cross-view linking: hand the current global pick to the fresh driver
      d.setSelection?.(appStore.getState().interpSelection);
      // Only spin a RAF for drivers that actually animate; static views (e.g.
      // the weight spectrum) redraw on demand and don't need a 60fps heartbeat.
      if (d.animated !== false) {
        const loop = (now: number) => {
          if (disposed) return;
          const dt = now - last;
          last = now;
          d.frame(dt, (now - t0) / 1000);
          raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
      }
    })();

    const ro = new ResizeObserver(() => {
      const sz = sizeNow();
      driverRef.current?.resize(sz.w, sz.h, dprOf());
    });
    ro.observe(host);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      if (driverRef.current) {
        driverRef.current.dispose();
        driverRef.current = null;
      }
    };
  }, [interp.featureId, model, tier, resolvedTrace]);

  // cross-view linking (2a): forward every selection change to the live driver
  const sel = $interpSelection.value;
  useEffect(() => {
    if (status.value === "ready") driverRef.current?.setSelection?.(sel);
  }, [sel]);

  // 1d — save the view's source bundle(s), straight from the fetch cache: one
  // file downloads verbatim; multiple are wrapped in a single JSON keyed by
  // filename, with enough meta to cite where the numbers came from.
  const downloadData = () => {
    const urls = dataUrls.value;
    if (!urls.length || !model || !feature) return;
    const fname = (u: string) => u.slice(u.lastIndexOf("/") + 1);
    let payload: unknown;
    let name: string;
    const only = urls.length === 1 ? urls[0] : undefined;
    if (only) {
      payload = cachedBundle(only);
      name = `${model}-${fname(only)}`;
    } else {
      payload = {
        meta: {
          app: "nebulai",
          feature: `#${feature.n} ${feature.label}`,
          model,
          sources: urls,
          saved: new Date().toISOString(),
        },
        files: Object.fromEntries(urls.map((u) => [fname(u), cachedBundle(u)])),
      };
      name = `nebulai-data-${feature.id}-${model}.json`;
    }
    if (payload === undefined) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(
      new Blob([JSON.stringify(payload)], { type: "application/json" }),
    );
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // 3a — guided tours. Store tracks only {id, step}; content lives in
  // chrome/tours.ts. Stepping applies (feature, trace, selection) through the
  // same store actions user clicks fire, so a step === a reachable app state.
  const tourRef = $tour.value;
  const activeTour = tourRef ? findTour(tourRef.id) : undefined;
  const tourStep = activeTour && tourRef ? activeTour.steps[tourRef.step] : undefined;
  const modelTours = TOURS.filter((t) => t.model === model);
  const exitTour = () => appStore.getState().setTour(null);
  const startTour = (id: string) => {
    const t = findTour(id);
    if (!t) return;
    appStore.getState().setTour({ id, step: 0 });
    applyTourStep(t, 0);
  };
  const gotoStep = (n: number) => {
    if (!activeTour || !tourRef || n < 0) return;
    if (n >= activeTour.steps.length) {
      exitTour(); // "finish" — leave the app exactly where the last step put it
      return;
    }
    appStore.getState().setTour({ id: tourRef.id, step: n });
    applyTourStep(activeTour, n);
  };

  // 2b — "+ your prompt": one real forward on the local live server, returned
  // in EXACTLY the offline trace bundle shape and injected into the bundle
  // cache, so every trace-driven view renders the typed prompt unchanged.
  const liveList = $liveTraces.value[model ?? ""] ?? [];
  const promptOpen = useSignal(false);
  const promptText = useSignal("");
  const promptBusy = useSignal(false);
  const promptErr = useSignal("");
  const runPrompt = async () => {
    const text = promptText.value.trim();
    if (!text || promptBusy.value || !model) return;
    promptBusy.value = true;
    promptErr.value = "";
    try {
      const base = $probing.value.liveUrl.replace(/\/+$/, "");
      let res: Response;
      try {
        res = await fetch(`${base}/live/trace`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
      } catch {
        throw new Error(
          `no live server at ${base} — start it: python -m nebulai.backend.interp.live_server`,
        );
      }
      const body = (await res.json()) as (TraceBundle & { truncated?: boolean }) | { error: string };
      if (!res.ok || "error" in body)
        throw new Error("error" in body ? body.error : `live server replied ${res.status}`);
      if (body.meta.model !== model)
        throw new Error(
          `the live server is running ${body.meta.model} but the viewer shows ${model} — ` +
            `restart it with --model ${model}`,
        );
      const slug =
        LIVE_TRACE_PREFIX +
        (body.meta.prompt
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-+|-+$/g, "")
          .slice(0, 40) || "prompt");
      putLiveTrace(model, slug, body);
      // 2c: the Piano-Roll re-derives SAE acts for this slug on demand via
      // POST /live/sae — it needs the prompt text behind the slug
      registerLivePrompt(model, slug, body.meta.prompt);
      $liveTraces.value = {
        ...$liveTraces.value,
        [model]: [
          ...liveList.filter((t) => t.slug !== slug),
          { slug, prompt: body.meta.prompt },
        ],
      };
      promptOpen.value = false;
      promptText.value = "";
      appStore.getState().setInterpTrace(slug);
    } catch (e) {
      promptErr.value = e instanceof Error ? e.message : String(e);
    } finally {
      promptBusy.value = false;
    }
  };

  // group the live features for the rail
  const byGroup = new Map<InterpGroup, typeof INTERP_FEATURES>();
  for (const f of INTERP_FEATURES) {
    const arr = byGroup.get(f.group) ?? [];
    arr.push(f);
    byGroup.set(f.group, arr);
  }

  return (
    <div class="interp-page" role="main">
      <aside class="interp-rail">
        <div class="interp-rail-head">
          <span class="interp-rail-title">Internals</span>
          <span class="interp-rail-count">{INTERP_FEATURES.length} of 25 live</span>
        </div>
        <div class="interp-model">
          <SelectRow
            label="Model"
            value={model ?? ""}
            disabled={$loading.value.active || $viewMode.value === "compare"}
            options={$datasets.value.map((d) => {
              const has = avail?.[d.id];
              return {
                value: d.id,
                label:
                  has === true ? `${d.id} ✓` : has === false ? `${d.id} — map only` : d.id,
              };
            })}
            onChange={(id) => requestDataset(id)}
          />
          <p class="interp-model-hint">
            {avail === null
              ? "checking which models have an internals export…"
              : model && avail[model]
                ? `✓ = has an internals export · reading out/${model}/interp`
                : "✓ = has an internals export — pick one, or export this model below"}
          </p>
        </div>
        {sel && (
          <div class="interp-linkpill" title="Cross-view selection — linked views highlight it">
            <span class="interp-linkpill-dot" />
            <span class="interp-linkpill-label">
              {sel.kind === "head"
                ? `head L${sel.layer}H${sel.head}`
                : sel.kind === "token"
                  ? `token pos ${sel.pos}`
                  : `SAE feature #${sel.id}`}
            </span>
            <button
              type="button"
              class="interp-linkpill-clear"
              aria-label="Clear cross-view selection"
              onClick={() => appStore.getState().setInterpSelection(null)}
            >
              ×
            </button>
          </div>
        )}
        <div class="interp-rail-scroll">
          <section class="interp-rail-group interp-tours">
            <h4 class="interp-rail-group-title">Guided tours</h4>
            {modelTours.length > 0 ? (
              modelTours.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  class={`interp-feature interp-tour-btn${tourRef?.id === t.id ? " is-active" : ""}`}
                  title={t.blurb}
                  onClick={() => startTour(t.id)}
                >
                  <span class="interp-feature-n">⚑</span>
                  <span class="interp-feature-label">{t.label}</span>
                </button>
              ))
            ) : (
              <p class="interp-tours-hint">
                tours quote gpt2 bundle numbers — switch to gpt2 to take one
              </p>
            )}
          </section>
          {[...byGroup.entries()].map(([group, feats]) => (
            <section key={group} class="interp-rail-group">
              <h4 class="interp-rail-group-title">{GROUP_LABEL[group]}</h4>
              {feats.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  class={`interp-feature${f.id === interp.featureId ? " is-active" : ""}`}
                  onClick={() => appStore.getState().setInterpFeature(f.id)}
                >
                  <span class="interp-feature-n">#{f.n}</span>
                  <span class="interp-feature-label">{f.label}</span>
                  {sel && f.linksTo?.includes(sel.kind) && (
                    <span
                      class="interp-feature-link"
                      title={`Highlights the selected ${sel.kind === "saeFeature" ? "SAE feature" : sel.kind}`}
                    />
                  )}
                </button>
              ))}
            </section>
          ))}
        </div>
        <p class="interp-rail-foot">
          Each view renders one real computed quantity. Hover any curve for exact
          values. Provenance and math on <span class="interp-kbd">/guide</span>.
        </p>
      </aside>

      <div class={`interp-stage${isForward && traces.value.length > 0 ? " has-tracebar" : ""}`}>
        {isForward && traces.value.length > 0 && (
          <div class="interp-tracebar" role="radiogroup" aria-label="Prompt">
            <span class="interp-tracebar-label">prompt</span>
            {[...traces.value, ...liveList].map((t) => (
              <button
                key={t.slug}
                type="button"
                role="radio"
                aria-checked={t.slug === resolvedTrace}
                class={`interp-trace${t.slug === resolvedTrace ? " is-active" : ""}${isLiveTrace(t.slug) ? " is-live" : ""}`}
                title={isLiveTrace(t.slug) ? `${t.prompt} — computed live this session` : t.prompt}
                onClick={() => appStore.getState().setInterpTrace(t.slug)}
              >
                {isLiveTrace(t.slug) && <span class="interp-trace-livedot" aria-hidden="true" />}
                {t.prompt}
              </button>
            ))}
            {promptOpen.value ? (
              <form
                class="interp-trace-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  void runPrompt();
                }}
              >
                <input
                  class="interp-trace-input"
                  type="text"
                  placeholder="type a prompt — one real forward on the live server"
                  value={promptText.value}
                  disabled={promptBusy.value}
                  ref={(el) => el?.focus()}
                  onInput={(e) => {
                    promptText.value = (e.target as HTMLInputElement).value;
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") {
                      promptOpen.value = false;
                      promptErr.value = "";
                    }
                  }}
                />
                <button type="submit" class="interp-trace" disabled={promptBusy.value}>
                  {promptBusy.value ? "running…" : "run ↵"}
                </button>
              </form>
            ) : (
              <button
                type="button"
                class="interp-trace interp-trace-add"
                title="Run the forward-trace views on your own prompt (needs the local live server)"
                onClick={() => {
                  promptOpen.value = true;
                  promptErr.value = "";
                }}
              >
                + your prompt
              </button>
            )}
            {promptErr.value && <span class="interp-tracebar-err">{promptErr.value}</span>}
          </div>
        )}

        {/* req 10 — keyboard focus + ARIA readout. The plot is a GPU canvas with
            no focusable DOM per datum, so the host itself is the focus target:
            tabbable, labelled with the live feature's name + blurb (composed
            from the registry, zero driver changes), and given a visible focus
            ring in CSS. Exact per-point values are announced by the driver's
            tooltip, which is an aria-live="polite" status region. */}
        <div
          class="interp-canvas-host"
          tabIndex={0}
          role="img"
          aria-label={
            feature
              ? `Chart #${feature.n}: ${feature.label}. ${feature.blurb}${
                  status.value === "ready" ? " Hover or focus a point for exact values." : ""
                }`
              : "Interpretability chart"
          }
        >
          <canvas ref={canvasRef} class="interp-canvas" />
          <div ref={overlayRef} class="interp-overlay" />
        </div>

        {activeTour && tourRef && tourStep && (
          <div class="interp-tourbar" role="group" aria-label={`Guided tour: ${activeTour.label}`}>
            <div class="interp-tourbar-head">
              <span class="interp-tourbar-tour">⚑ {activeTour.label}</span>
              <span class="interp-tourbar-count">
                {tourRef.step + 1} / {activeTour.steps.length}
              </span>
              <button
                type="button"
                class="interp-tourbar-exit"
                aria-label="Exit tour"
                onClick={exitTour}
              >
                ×
              </button>
            </div>
            <h4 class="interp-tourbar-title">{tourStep.title}</h4>
            <p class="interp-tourbar-caption">{tourStep.caption}</p>
            <div class="interp-tourbar-nav">
              <button
                type="button"
                disabled={tourRef.step === 0}
                onClick={() => gotoStep(tourRef.step - 1)}
              >
                ‹ back
              </button>
              <button
                type="button"
                class="is-primary"
                onClick={() => gotoStep(tourRef.step + 1)}
              >
                {tourRef.step + 1 >= activeTour.steps.length ? "finish ✓" : "next ›"}
              </button>
            </div>
          </div>
        )}

        {feature && (
          <div
            class={`interp-legend corner-${feature.legendCorner ?? "tr"}${legendIsOpen ? "" : " is-collapsed"}`}
          >
            <div class="interp-legend-head">
              <span class="interp-legend-n">#{feature.n}</span>
              <h3 class="interp-legend-title">{feature.label}</h3>
              {legendIsOpen && <span class="interp-legend-model">{model ?? "—"}</span>}
              {status.value === "ready" && dataUrls.value.length > 0 && (
                <button
                  type="button"
                  class="interp-legend-toggle interp-legend-data"
                  title={`Download this view's data (${dataUrls.value.length} file${dataUrls.value.length > 1 ? "s" : ""})`}
                  aria-label="Download this view's source data as JSON"
                  onClick={downloadData}
                >
                  ⤓
                </button>
              )}
              <button
                type="button"
                class="interp-legend-toggle"
                aria-expanded={legendIsOpen}
                title={legendIsOpen ? "Collapse legend" : "Expand legend"}
                onClick={() => {
                  legendOpen.value = !legendIsOpen;
                }}
              >
                {legendIsOpen ? "−" : "+"}
              </button>
            </div>
            {legendIsOpen && <p class="interp-legend-blurb">{feature.blurb}</p>}
            {legendIsOpen && feature.legend && (
              <ul class="interp-legend-keys">
                {feature.legend.map((k) => (
                  <li key={k.label}>
                    <span
                      class="interp-legend-swatch"
                      style={{ background: `rgb(${k.rgb})` }}
                    />
                    {k.label}
                  </li>
                ))}
                {feature.note && <li class="interp-legend-note">{feature.note}</li>}
              </ul>
            )}
          </div>
        )}

        {$loading.value.active ? (
          <div class="interp-status is-loading">
            <strong>Switching model…</strong>
            <span>
              {$loading.value.total > 0
                ? `downloading dataset — ${($loading.value.loaded / 1e6).toFixed(1)} / ${($loading.value.total / 1e6).toFixed(1)} MB`
                : "downloading dataset…"}
            </span>
            <span class="interp-status-bar">
              <span
                class="interp-status-bar-fill is-measured"
                style={{
                  width: `${$loading.value.total > 0 ? Math.min(100, ($loading.value.loaded / $loading.value.total) * 100) : 0}%`,
                }}
              />
            </span>
            <span class="interp-status-hint">
              the internals view refreshes automatically when the model lands
            </span>
          </div>
        ) : status.value === "loading" ? (
          <div class="interp-status is-loading">
            <strong>{feature ? `#${feature.n} ${feature.label}` : "Loading view"}</strong>
            <span>
              {phase.value === "renderer"
                ? "starting the renderer…"
                : `fetching ${model} bundle + computing…`}
            </span>
            <span class="interp-status-bar">
              <span class="interp-status-bar-fill" />
            </span>
            <span class="interp-status-hint">
              first open fetches out/{model}/interp — cached after that
            </span>
          </div>
        ) : null}
        {status.value === "error" && !$loading.value.active && isLiveTrace(resolvedTrace) ? (
          // the failure is about the typed prompt, not a missing export — say
          // so, and offer the way back to the bundled prompts that DO work
          <div class="interp-status is-error">
            <strong>“{feature?.label ?? interp.featureId}” can’t run on a custom prompt.</strong>
            <span>{errMsg.value}</span>
            {traces.value.length > 0 && (
              <div class="interp-status-actions">
                <button
                  type="button"
                  class="interp-status-switch"
                  onClick={() => appStore.getState().setInterpTrace(traces.value[0]?.slug ?? "")}
                >
                  back to “{traces.value[0]?.prompt}”
                </button>
              </div>
            )}
          </div>
        ) : status.value === "error" && !$loading.value.active && (
          <div class="interp-status is-error">
            {hasIdx.value ? (
              // the export exists — this specific bundle just isn't in it, so
              // re-running the CLI wouldn't help. Say why honestly.
              <>
                <strong>
                  “{feature?.label ?? interp.featureId}” isn’t in {model}’s internals export.
                </strong>
                <span>
                  {feature?.group === "sae"
                    ? "SAE views are computed from the res-jb SAE release, which only covers gpt2."
                    : feature?.group === "trained"
                      ? "The grokking toy-model bundle ships with the gpt2 export only."
                      : errMsg.value}
                </span>
              </>
            ) : (
              <>
                <strong>No internals export for “{model}”.</strong>
                <span>{errMsg.value}</span>
              </>
            )}
            {avail && (
              <div class="interp-status-actions">
                {$datasets.value
                  .filter((d) => d.id !== model && avail[d.id])
                  // gpt2-gated bundles: pointing at another SAE-less model
                  // would only reproduce the failure
                  .filter(
                    (d) =>
                      !(
                        hasIdx.value &&
                        (feature?.group === "sae" || feature?.group === "trained")
                      ) || d.id === "gpt2",
                  )
                  .map((d) => (
                    <button
                      key={d.id}
                      type="button"
                      class="interp-status-switch"
                      onClick={() => requestDataset(d.id)}
                    >
                      switch to {d.id}
                    </button>
                  ))}
              </div>
            )}
            {hasIdx.value !== true && (
              <span class="interp-status-hint">
                or export this model:{" "}
                <span class="interp-kbd">nebulai interp --model {model}</span>
              </span>
            )}
          </div>
        )}
        {!model && (
          <div class="interp-status">Load a dataset to inspect its internals.</div>
        )}
      </div>
    </div>
  );
}
