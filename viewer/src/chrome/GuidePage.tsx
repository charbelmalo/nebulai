/** Guide — the math + provenance behind every live Internals view. Reads the
 *  SAME registry the Internals rail does (INTERP_FEATURES), so the documentation
 *  can never drift from what's actually shipped: a feature that isn't live can't
 *  appear here, and a live feature must carry its `math` and `source` (both
 *  required on InterpFeature) to compile. Each card links straight into the live
 *  view so the reader can check the numbers themselves. */

import { appStore } from "../app/store";
import type { InterpGroup } from "../scene/interp/InterpDriver";
import { GROUP_LABEL, INTERP_FEATURES } from "../scene/interp/registry";

const GROUP_ORDER: InterpGroup[] = ["weights", "forward", "sae", "trained", "live"];

const GROUP_SOURCE: Record<InterpGroup, string> = {
  weights: "Raw weight tensors only — no forward pass. Computed offline in float64.",
  forward: "One real forward pass on a curated prompt (pick the prompt in Internals).",
  sae: "Sparse-autoencoder features from downloaded SAE weights.",
  trained: "A small model trained offline (e.g. a grokking toy model).",
  live:
    "A real forward pass on text you type — computed on request by a local " +
    "probe server running the same numpy GPT-2 as every offline bundle " +
    "(weights stay on your machine; nothing is precomputed).",
};

function openInInternals(id: string): void {
  const s = appStore.getState();
  s.setInterpFeature(id);
  s.setPage("interp");
}

export function GuidePage() {
  const live = INTERP_FEATURES.length;
  const byGroup = new Map<InterpGroup, typeof INTERP_FEATURES>();
  for (const f of INTERP_FEATURES) {
    const arr = byGroup.get(f.group) ?? [];
    arr.push(f);
    byGroup.set(f.group, arr);
  }

  return (
    <div class="guide-page" role="main">
      <div class="guide-scroll">
        <header class="guide-head">
          <p class="guide-kicker">Nebul.AI · Internals</p>
          <h1 class="guide-title">The math behind every view</h1>
          <p class="guide-lede">
            Each Internals view renders exactly <em>one real computed quantity</em>{" "}
            from a model — attention, residual norms, singular-value spectra, a
            positional-embedding DFT, the logit lens. No placeholder data, no fake
            motion, no misleading encodings: the axes and colors <em>are</em> the
            numbers, and every view exposes exact hover values. This page states
            the precise formula and the on-disk source for each one, so the map is
            defensible — and, where a view has a known honest artifact (an
            attention sink, a massive activation), it says so.
          </p>
          <p class="guide-count">
            <strong>{live} of 25</strong> features are live. A feature only appears
            once its driver renders real data end-to-end; the rest await their
            source data (SAE weights, gradients, a trained toy model, a live
            forward pass) and are deliberately not shown.
          </p>
        </header>

        {GROUP_ORDER.filter((g) => byGroup.has(g)).map((group) => (
          <section key={group} class="guide-group">
            <div class="guide-group-head">
              <h2 class="guide-group-title">{GROUP_LABEL[group]}</h2>
              <p class="guide-group-src">{GROUP_SOURCE[group]}</p>
            </div>
            <div class="guide-cards">
              {byGroup.get(group)!.map((f) => (
                <article key={f.id} class="guide-card">
                  <div class="guide-card-head">
                    <span class="guide-card-n">#{f.n}</span>
                    <h3 class="guide-card-label">{f.label}</h3>
                    <button
                      type="button"
                      class="guide-card-open"
                      onClick={() => openInInternals(f.id)}
                    >
                      Open in Internals →
                    </button>
                  </div>
                  <p class="guide-card-blurb">{f.blurb}</p>
                  <div class="guide-card-row">
                    <span class="guide-card-tag">math</span>
                    <code class="guide-card-math">{f.math}</code>
                  </div>
                  <div class="guide-card-row">
                    <span class="guide-card-tag">source</span>
                    <span class="guide-card-source">{f.source}</span>
                  </div>
                  {f.legend && (
                    <ul class="guide-card-legend">
                      {f.legend.map((k) => (
                        <li key={k.label}>
                          <span
                            class="guide-card-swatch"
                            style={{ background: `rgb(${k.rgb})` }}
                          />
                          {k.label}
                        </li>
                      ))}
                    </ul>
                  )}
                </article>
              ))}
            </div>
          </section>
        ))}

        <footer class="guide-foot">
          <p>
            Reproduce these bundles with{" "}
            <span class="interp-kbd">nebulai interp --model &lt;id&gt;</span> — it
            runs the forward pass, captures attention / residual norms / the logit
            lens, and SVD-decomposes the weights, writing{" "}
            <span class="interp-kbd">out/&lt;id&gt;/interp/*.json</span>. The viewer
            reads exactly those files; nothing here is synthesized in the browser
            beyond the stated transforms (DFT, SVD summaries, attention rollout).
          </p>
        </footer>
      </div>
    </div>
  );
}
