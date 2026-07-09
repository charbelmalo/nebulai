/** Internals — the mechanistic-interpretability gallery. A dedicated page with
 *  its own canvas, RAF loop, and feature rail. One InterpDriver owns the canvas
 *  at a time; each renders exactly ONE real computed quantity from an interp
 *  bundle (out/<model>/interp/*.json) and exposes exact hover values. The rail
 *  only lists features whose driver is live and backed by real data — an
 *  unimplemented feature never appears, so the page can't overstate what it shows.
 *
 *  The active model is the current dataset id (bundles are per-model). If a model
 *  has no interp export, the driver's setModel rejects and we say so plainly. */

import { useSignal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";
import { appStore } from "../app/store";
import { loadInterpIndex } from "../data/interp";
import type { InterpDriver, InterpGroup } from "../scene/interp/InterpDriver";
import { GROUP_LABEL, INTERP_FEATURES, findFeature } from "../scene/interp/registry";
import { $capabilities, $datasetId, $interp } from "./state";

interface TraceEntry {
  slug: string;
  prompt: string;
}

export function InterpPage() {
  const interp = $interp.value;
  const model = $datasetId.value;
  const caps = $capabilities.value;
  const tier = caps?.tier ?? "webgl";
  const feature = findFeature(interp.featureId);

  const status = useSignal<"loading" | "ready" | "error">("loading");
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
  const isForward = feature?.group === "forward" || feature?.perTrace === true;
  useEffect(() => {
    if (!model) return;
    let ok = true;
    loadInterpIndex(model)
      .then((idx) => ok && (traces.value = idx.traces ?? []))
      .catch(() => ok && (traces.value = []));
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
    errMsg.value = "";

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
      try {
        await d.setModel(model, resolvedTrace);
      } catch (e) {
        if (disposed) {
          driverRef.current?.dispose();
          driverRef.current = null;
          return;
        }
        status.value = "error";
        errMsg.value = e instanceof Error ? e.message : String(e);
        return;
      }
      if (disposed) {
        driverRef.current?.dispose();
        driverRef.current = null;
        return;
      }
      status.value = "ready";
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
        <div class="interp-rail-scroll">
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

      <div class="interp-stage">
        <canvas ref={canvasRef} class="interp-canvas" />
        <div ref={overlayRef} class="interp-overlay" />

        {isForward && traces.value.length > 0 && (
          <div class="interp-tracebar" role="radiogroup" aria-label="Prompt">
            <span class="interp-tracebar-label">prompt</span>
            {traces.value.map((t) => (
              <button
                key={t.slug}
                type="button"
                role="radio"
                aria-checked={t.slug === resolvedTrace}
                class={`interp-trace${t.slug === resolvedTrace ? " is-active" : ""}`}
                title={t.prompt}
                onClick={() => appStore.getState().setInterpTrace(t.slug)}
              >
                {t.prompt}
              </button>
            ))}
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

        {status.value === "loading" && (
          <div class="interp-status">computing…</div>
        )}
        {status.value === "error" && (
          <div class="interp-status is-error">
            <strong>No interp bundle for “{model}”.</strong>
            <span>{errMsg.value}</span>
            <span class="interp-status-hint">
              Run <span class="interp-kbd">nebulai interp --model {model}</span> to
              export it.
            </span>
          </div>
        )}
        {!model && (
          <div class="interp-status">Load a dataset to inspect its internals.</div>
        )}
      </div>
    </div>
  );
}
