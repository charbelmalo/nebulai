/** Guide — the math + provenance behind every live Internals view. Reads the
 *  SAME registry the Internals rail does (INTERP_FEATURES), so the documentation
 *  can never drift from what's actually shipped: a feature that isn't live can't
 *  appear here, and a live feature must carry its `math` and `source` (both
 *  required on InterpFeature) to compile. Each card links straight into the live
 *  view so the reader can check the numbers themselves. */

import { appStore } from "../app/store";
import type { GuideFormula, InterpGroup } from "../scene/interp/InterpDriver";
import { GROUP_LABEL, INTERP_FEATURES } from "../scene/interp/registry";
import { GUIDE_RESEARCH } from "./guideResearch";

const GROUP_ORDER: InterpGroup[] = ["weights", "forward", "sae", "trained", "live"];

const GROUP_SOURCE: Record<InterpGroup, string> = {
  weights:
    "Measurements taken directly from the model’s stored parameters. The model did not " +
    "process a prompt for these views. Calculations were run offline at high precision.",
  forward:
    "Measurements from one complete model run on a prepared prompt. Choose the prompt " +
    "on the Internals page.",
  sae:
    "Patterns found by a sparse autoencoder (SAE), a separate tool that breaks model " +
    "activity into smaller, reusable features. These views use downloaded SAE parameters.",
  trained:
    "Measurements from a small model trained offline for a focused experiment, such as " +
    "modular addition. It is not GPT-2 unless stated.",
  live:
    "Measurements from a complete model run on text you enter. A local server performs " +
    "the calculation by default, so your prompt and model weights stay on your machine. " +
    "If you choose a remote server in Settings, your prompt is sent there instead.",
};

function openInInternals(id: string): void {
  const s = appStore.getState();
  s.setInterpFeature(id);
  s.setPage("interp");
}

/** MathML has native layout support in every browser we support. Formula markup
 * comes only from the static feature registry, never from prompts or fetched
 * data; the matching aria label provides a usable spoken equivalent. */
function GuideFormulaView({ formula }: { formula: GuideFormula }) {
  return (
    <math
      class="guide-card-formula"
      aria-label={formula.ariaLabel}
      role="math"
      display="block"
      dangerouslySetInnerHTML={{ __html: formula.mathml }}
    />
  );
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
          <p class="guide-kicker">Nebul.AI · Model Guide</p>
          <h1 class="guide-title">How to read every model view</h1>
          <p class="guide-lede">
            Each view shows one measurement taken from a model. Hover to inspect exact
            values. This guide explains what the view measures, how to read its colors
            and axes, how the numbers were calculated, and where the data came from.
            When a view has an important limitation or known artifact, we call it out.
          </p>
          <p class="guide-count">
            <strong>{live} of 25</strong> planned views are available. We publish a
            view only after it works from source data to visualization. Views that
            still need data or computation stay hidden until they are ready.
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
                      Explore this view →
                    </button>
                  </div>
                  <p class="guide-card-blurb">{f.blurb}</p>
                  <div class="guide-card-row">
                    <span class="guide-card-tag">Calculation</span>
                    <div class="guide-card-calculation">
                      <p class="guide-card-math">{f.math}</p>
                      {f.formulas?.map((formula, i) => (
                        <GuideFormulaView key={`${f.id}-formula-${i}`} formula={formula} />
                      ))}
                    </div>
                  </div>
                  <div class="guide-card-row">
                    <span class="guide-card-tag">Data source</span>
                    <span class="guide-card-source">{f.source}</span>
                  </div>
                  <div class="guide-card-row">
                    <span class="guide-card-tag">Research</span>
                    <ol class="guide-card-research">
                      {GUIDE_RESEARCH[f.id].map((source) => (
                        <li key={source.url}>
                          <a href={source.url} target="_blank" rel="noreferrer">
                            {source.title}
                          </a>
                          <span>{source.citation}</span>
                        </li>
                      ))}
                    </ol>
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
            To rebuild the data, run{" "}
            <span class="interp-kbd">nebulai interp --model &lt;id&gt;</span>. This runs
            the model, records the measurements used by the views, and saves them
            under <span class="interp-kbd">out/&lt;id&gt;/interp/*.json</span>. The
            browser reads those files and only applies the transformations named in
            this guide.
          </p>
        </footer>
      </div>
    </div>
  );
}
