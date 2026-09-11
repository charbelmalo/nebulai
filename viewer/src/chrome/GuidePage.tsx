/** Guide — the math + provenance behind every live Internals view. Reads the
 *  SAME registry the Internals rail does (INTERP_FEATURES), so the documentation
 *  can never drift from what's actually shipped: a feature that isn't live can't
 *  appear here, and a live feature must carry its `math` and `source` (both
 *  required on InterpFeature) to compile. Each card links straight into the live
 *  view so the reader can check the numbers themselves. */

import { requestEpisodeStep } from "../app/actions";
import { appStore } from "../app/store";
import { $channels, channelsFor, channelsLoaded, ensureChannels } from "../data/channels";
import {
  $directions,
  directionsFor,
  directionsLoaded,
  ensureDirections,
} from "../data/directions";
import type { GuideFormula, InterpGroup } from "../scene/interp/InterpDriver";
import { GROUP_LABEL, INTERP_FEATURES } from "../scene/interp/registry";
import { guideResearchFor } from "./guideResearch";
import { $datasets } from "./state";
import { episodeAvailability, TOURS, type EpisodeContext } from "./tours";

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

/* ── episodes (P5) ────────────────────────────────────────────────────────── */

/** The episode list, gated on what this deploy can actually show.
 *
 *  The gate is the whole reason this section exists rather than a row of
 *  buttons. Every episode quotes exact numbers out of one named artifact. If
 *  that artifact is not here, there are three honest answers and this renders
 *  all three differently:
 *
 *  · **ready** — the map, the channels and the views the captions point at are
 *    all present. The button plays it.
 *  · **pending** — the sidecar is being fetched right now. Not an error, and
 *    not a promise either; it says what it is waiting on.
 *  · **unavailable** — something is genuinely absent. The card stays, the
 *    button is disabled, and the card prints WHICH file and WHICH command
 *    would produce it. It never falls back to another model, and it never
 *    plays with the numbers missing (§2.2).
 */
function EpisodeSection() {
  const entries = $datasets.value;
  // touching the signal here is what subscribes this component to the fetch
  // resolving, so a "pending" card becomes a "ready" one without a click
  void $channels.value;
  void $directions.value;

  // kick off the sidecar fetch for every dataset an episode names, with the
  // point count the index already knows — the same expected length the map
  // itself checks with, so a channels.json aligned to a different build is
  // rejected here exactly as it would be there
  for (const t of TOURS) {
    const dsId = t.manifest?.dataset ?? (t.manifest?.channels?.length ? t.model : null);
    if (!dsId) continue;
    const entry = entries.find((e) => e.id === dsId);
    if (entry) ensureChannels(dsId, entry.n_points);
  }
  // and the direction sidecar for every episode that names one. Separate loop
  // because an episode may name directions without naming channels — the
  // refusal-style one does exactly that, since its direction is in resid.L8
  // and therefore has no channels on this map at all.
  for (const t of TOURS) {
    const dsId = t.manifest?.directions?.length ? (t.manifest.dataset ?? t.model) : null;
    if (dsId && entries.some((e) => e.id === dsId)) ensureDirections(dsId);
  }

  const ctx: EpisodeContext = {
    datasets: entries.map((e) => e.id),
    channelsFor: (id) => channelsFor(id)?.channels.map((c) => c.id) ?? null,
    channelsLoaded: (id) => channelsLoaded(id),
    directionsFor: (id) => directionsFor(id)?.directions.map((d) => d.id) ?? null,
    directionsLoaded: (id) => directionsLoaded(id),
    features: INTERP_FEATURES.map((f) => f.id),
  };

  return (
    <section class="guide-group guide-episodes">
      <div class="guide-group-head">
        <h2 class="guide-group-title">Episodes</h2>
        <p class="guide-group-src">
          Guided walks through one finding at a time. Each one quotes exact numbers from
          one named artifact and says which; if that artifact is not in this deploy, the
          episode says so rather than running with the numbers missing.
        </p>
      </div>
      <div class="guide-cards">
        {TOURS.map((t) => {
          const av = episodeAvailability(t, ctx);
          const m = t.manifest;
          return (
            <article key={t.id} class={`guide-card episode-card is-${av.state}`}>
              <div class="guide-card-head">
                <span class="guide-card-n">{t.steps.length} steps</span>
                <h3 class="guide-card-label">{t.label}</h3>
                <button
                  type="button"
                  class="guide-card-open"
                  disabled={av.state !== "ready"}
                  onClick={() => requestEpisodeStep(t.id, 0)}
                >
                  Play this episode →
                </button>
              </div>
              <p class="guide-card-blurb">{t.blurb}</p>
              <div class="guide-card-row">
                <span class="guide-card-tag">Model</span>
                <span class="guide-card-source">
                  {t.model}
                  {m?.space ? ` · ${m.space}` : ""}
                </span>
              </div>
              {m?.channels?.length ? (
                <div class="guide-card-row">
                  <span class="guide-card-tag">Channels</span>
                  <span class="guide-card-source">{m.channels.join(", ")}</span>
                </div>
              ) : null}
              {m?.directions?.length ? (
                <div class="guide-card-row">
                  <span class="guide-card-tag">Directions</span>
                  <span class="guide-card-source">{m.directions.join(", ")}</span>
                </div>
              ) : null}
              {av.state !== "ready" && (
                <p class={`episode-gate is-${av.state}`}>
                  {av.state === "pending" ? av.reason : `Not available here — ${av.reason}`}
                </p>
              )}
            </article>
          );
        })}
      </div>
    </section>
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

        <EpisodeSection />

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
                      {guideResearchFor(f.id).map((source) => (
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
