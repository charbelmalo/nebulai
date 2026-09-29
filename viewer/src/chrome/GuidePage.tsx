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
import { APP_ROOT } from "../data/base";
import { experienceHref } from "./ExperienceNav";
import { guideResearchFor } from "./guideResearch";
import { $datasetId, $datasets, $experience } from "./state";
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
/** An episode's address in Learn: guided walks run there, whatever page a
 *  step borrows, so Research's Methods links across instead of playing it. */
export function learnEpisodeHref(id: string): string {
  return `${new URL("learn/", APP_ROOT).href}#episode=${encodeURIComponent(id)}&step=0`;
}

function EpisodeSection({ mode }: { mode: "play" | "handoff" }) {
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
        <h2 class="guide-group-title">{mode === "play" ? "Guided episodes" : "Episodes"}</h2>
        <p class="guide-group-src">
          Guided walks through one finding at a time. Each one quotes exact numbers from
          one named artifact and says which; if that artifact is not in this deploy, the
          episode says so rather than running with the numbers missing.
          {mode === "handoff" && " Episodes play in NebulAI Learn, which keeps the step controls with you."}
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
                {mode === "handoff" && av.state === "ready" ? (
                  <a class="guide-card-open" href={learnEpisodeHref(t.id)}>
                    Play in Learn ↗
                  </a>
                ) : (
                  <button
                    type="button"
                    class="guide-card-open"
                    disabled={av.state !== "ready"}
                    onClick={() => requestEpisodeStep(t.id, 0)}
                  >
                    Play this episode →
                  </button>
                )}
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

/* ── the claim contract (§2.4 / D3) ───────────────────────────────────────── */

/** The one causal sentence this project permits, and the boundary around it.
 *
 *  Every other view on the Internals page measures a model that was left alone,
 *  and the honesty rule for those is flat: no causal claims. A view that
 *  installs a hook is the single exception, and it is a narrow one — the
 *  sentence may say what the intervention DID, under its protocol, with its
 *  control, and may not say what the direction or feature IS. Both halves are
 *  printed, because the permission is worthless without the prohibition
 *  attached to it.
 *
 *  It renders on the card of every feature whose registry entry sets
 *  `intervenes`, rather than once in the page footer. A footer is the part of a
 *  page that does not travel: the number gets screenshotted, quoted and
 *  forwarded, and the contract has to be inside the crop. */
function ClaimContract() {
  return (
    <div class="guide-card-row guide-card-claim">
      <span class="guide-card-tag">Claims</span>
      <div class="guide-card-claimbody">
        <p class="guide-card-claimlede">
          This view changes the model's forward pass, so it is allowed one causal
          sentence — of exactly this shape, and no other:
        </p>
        <p class="guide-card-claimquote">
          Under protocol P, intervening on direction <em>d</em> at layer L changed
          behaviour B from X to Y (n = …, seed = …).
        </p>
        <p class="guide-card-claimnote">
          What that sentence does <strong>not</strong> say is what <em>d</em> is.
          It is not the refusal direction, the truth direction or the Golden Gate
          feature: it is a direction extracted by a named method from named data
          which, when intervened on, moved a measured behaviour. Every figure here
          ships with the α = 0 control that installed no hook, and the control is
          drawn rather than assumed. No weights are modified or written.
        </p>
      </div>
    </div>
  );
}

/** Learn's catalog: lessons and guided episodes only. The per-view method
 *  cards open unguided Internals analysis, which belongs to Research, so here
 *  they are one explicit link away rather than 26 equal choices. */
function LessonsPage() {
  const datasetId = $datasetId.value;
  return (
    <div class="guide-page" role="main">
      <div class="guide-scroll">
        <header class="guide-head">
          <p class="guide-kicker">NebulAI Learn · Lessons</p>
          <h1 class="guide-title">How model maps work</h1>
          <p class="guide-lede">
            A model map places things a model has learned — words, directions, features —
            so that related ones sit near each other. Each lesson shows one idea on real
            published data and says what the map cannot tell you. You can stop a lesson at
            any step.
          </p>
        </header>

        <EpisodeSection mode="play" />

        <section class="guide-group guide-next" aria-labelledby="guide-next-title">
          <div class="guide-group-head">
            <h2 class="guide-group-title" id="guide-next-title">
              Where next
            </h2>
          </div>
          <div class="guide-next-links">
            <a class="guide-next-link" href={experienceHref("atlas", "learn", datasetId)}>
              <strong>Continue in Atlas</strong>
              <span>Search the full map on your own, inspect a unit and save an exact record.</span>
            </a>
            <a class="guide-next-link" href={`${new URL("research/", APP_ROOT).href}#page=guide`}>
              <strong>Read the methods in Research</strong>
              <span>How each of the registered analyses is calculated, with its sources.</span>
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}

export function GuidePage() {
  if ($experience.value === "learn") return <LessonsPage />;
  return <MethodsPage />;
}

/** Research's Methods: every registered view's calculation, data source and
 *  references, plus the episode list handing off to Learn. */
function MethodsPage() {
  const live = INTERP_FEATURES.length;
  // the roadmap in docs/INTERP_FEATURES.md planned 25 views, all of which ship;
  // #26 is the intervention rail, added later by ATTRACTORS-PLAN phase 4. Both
  // numbers are derived, so neither can drift from what is actually registered.
  const planned = 25;
  const extra = live - planned;
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
          <p class="guide-kicker">NebulAI Research · Methods</p>
          <h1 class="guide-title">How to read every model view</h1>
          <p class="guide-lede">
            Each view shows one measurement taken from a model. Hover to inspect exact
            values. This guide explains what the view measures, how to read its colors
            and axes, how the numbers were calculated, and where the data came from.
            When a view has an important limitation or known artifact, we call it out.
          </p>
          <p class="guide-count">
            <strong>
              {Math.min(live, planned)} of {planned}
            </strong>{" "}
            planned views are available
            {extra > 0 && (
              <>
                , plus {extra} added since: the intervention rail, which changes the
                model's forward pass instead of only measuring it
              </>
            )}
            . We publish a view only after it works from source data to
            visualization. Views that still need data or computation stay hidden
            until they are ready.
          </p>
        </header>

        <EpisodeSection mode={$experience.value === null ? "play" : "handoff"} />

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
                  {f.intervenes && <ClaimContract />}
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
