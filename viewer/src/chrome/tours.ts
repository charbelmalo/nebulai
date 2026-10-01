/** Guided story tours (3a) — scripted walks through findings that already
 *  ship in the interp bundles. A tour step is nothing more than a saved
 *  (feature, trace, cross-view selection) plus a caption: stepping applies it
 *  through the SAME store actions a user's clicks would fire, so every step is
 *  also a permalinkable app state, and exiting a tour leaves the app exactly
 *  where the last step put it.
 *
 *  Honesty rule for captions: every number quoted below is read from the
 *  shipped gpt2 bundles (out/gpt2/interp/*.json) — nothing is narrated that
 *  the view on screen can't corroborate. Tours are gpt2-only because the
 *  findings are: the induction/IOI/SAE stories were computed on gpt2.
 */

import type { Selection as MapSelection } from "../app/slices/atlas";
import { appStore, type InterpSelection } from "../app/store";
import { selectSteer, type SteerCell } from "../scene/interp/steer";

export interface TourStep {
  /** registry feature id to show. Omit for a step that lives on the Map page. */
  feature?: string;
  /** trace slug for per-trace features; omit for ownPrompts views */
  trace?: string;
  /** cross-view selection to pin (applied via setInterpSelection) */
  selection?: InterpSelection;
  /** which (alpha, prompt) cell of an intervention sweep this step is about.
   *  Applied through the same `selectSteer` the slider calls, so an episode
   *  step and a user drag leave the app in identical states — and so a caption
   *  that quotes a cell's numbers is standing next to that cell. */
  steer?: SteerCell;

  /* ── map-page steps (P5 episodes) ─────────────────────────────────────── */
  /** which page this step is told on; defaults to "interp" */
  page?: "map" | "interp";
  /** dataset id to load first. A step may change model mid-episode. */
  dataset?: string;
  /** channel id to light the lens with, or null to put the lens away */
  channel?: string | null;
  /** filter window in the channel's RAW units — the numbers the caption quotes */
  channelWindow?: [number, number] | null;
  /** keyword search to run on the map (the labels the caption names) */
  search?: string;
  /** map-page pick to pin — a cluster or a point index in THIS dataset */
  mapSelection?: MapSelection | null;
  /** direction id to lay the map out on, or null to put the axis away. Same
   *  three-state convention as `channel`: `undefined` leaves it alone. */
  axis?: string | null;
  /** how far onto that axis, 0–1. Ignored unless `axis` is set. */
  axisT?: number;

  /** A lesson step the reader has to DO, not watch. Only lessons carry one;
   *  the runner never marks such a step complete because it was shown. */
  task?: LessonTask;

  title: string;
  caption: string;
}

/** What a lesson step asks of the reader. Pure data: whether it is done is
 *  decided by chrome/learn/lesson.ts from the reader's actual selection and
 *  answers, never from a timer or an animation finishing. */
export type LessonTask =
  /** select one unit whose own label matches `match` (case-insensitive) */
  | { kind: "find"; match: string; hint: string }
  /** a qualitative question; only the correct option completes the step */
  | { kind: "check"; question: string; options: LessonOption[] }
  /** save the unit, continue in Atlas, or finish explicitly */
  | { kind: "finish" };

export interface LessonOption {
  id: string;
  text: string;
  correct: boolean;
  /** shown after the option is chosen, right or wrong */
  why: string;
}

/** What an episode needs in order to be told truthfully.
 *
 *  An episode is a claim about a specific artifact: "these eleven tokens sit
 *  1.53 from the raw centroid" is true of `out/gpt2/channels.json` and of
 *  nothing else. A deploy that ships without that file must therefore not
 *  render the episode with the numbers missing, and must not quietly substitute
 *  another model — it must say what is absent and why (§2.2: a pinned model id
 *  is never substituted, and absence is "not measured").
 *
 *  Every field is a hard requirement, checked against what the app can actually
 *  see before the episode is offered. */
export interface TourManifest {
  /** map dataset id that must be present in `out/index.json` */
  dataset?: string;
  /** channel ids that must exist in that dataset's `channels.json` */
  channels?: string[];
  /** direction ids that must exist in that dataset's `directions.json`.
   *
   *  Checked for PRESENCE, not for renderability. An episode may legitimately
   *  be about a direction that cannot be drawn — the refusal-style one below
   *  is exactly that — and gating such an episode on the very property it
   *  exists to explain would delete the explanation along with the figure. */
  directions?: string[];
  /** Internals feature ids that must be live in the registry */
  features?: string[];
  /** the space tag every quantity in this episode lives in (see spaces.py).
   *  Recorded rather than checked: it is what stops a later step from plotting
   *  this episode's numbers against a quantity from a different basis. */
  space?: string;
  /** printed verbatim beside the episode when a requirement is missing, to say
   *  what would have to exist for it to run */
  unavailable?: string;
}

export interface Tour {
  id: string;
  /** "lesson" = an introductory lesson with reader tasks (chrome/learn);
   *  omitted = a narrated episode */
  kind?: "lesson";
  label: string;
  blurb: string;
  /** tours quote bundle-specific numbers — only offered on this model */
  model: string;
  steps: TourStep[];
  manifest?: TourManifest;
}

const IOI = "when-mary-and-john-went-to-the-store-joh";
const EIFFEL = "the-eiffel-tower-is-located-in-the-city-";

export const TOURS: Tour[] = [
  {
    id: "induction",
    label: "The Induction Circuit",
    blurb: "From raw weights to a verified two-head circuit — predicted, confirmed, ablated.",
    model: "gpt2",
    steps: [
      {
        feature: "comp-web",
        selection: { kind: "head", layer: 4, head: 11 },
        title: "Weights predict a circuit",
        caption:
          "No forward pass yet — just weight algebra. L4H11's output composes into the " +
          "KEYS of later heads: its top K-composition targets are L6H9 (0.103), L5H1 " +
          "(0.102) and L5H5 (0.097). Remember those names.",
      },
      {
        feature: "head-fingerprints",
        selection: { kind: "head", layer: 4, head: 11 },
        title: "L4H11 is the previous-token head",
        caption:
          "On the diagnostic pass, L4H11 attends to position t−1 with score 0.9996 — " +
          "essentially perfect. Induction needs exactly this ingredient: some head must " +
          "write “what came before me” into each position.",
      },
      {
        feature: "induction-microscope",
        selection: { kind: "head", layer: 5, head: 5 },
        title: "Behavior confirms the prediction",
        caption:
          "Feed 48 random tokens repeated twice: the top induction scorers are L7H10 " +
          "(0.92), L5H5 (0.90), L6H9 (0.90) and L5H1 (0.89) — the very heads L4H11's " +
          "K-composition pointed at, found here by behavior alone.",
      },
      {
        feature: "ablation-ghosts",
        selection: { kind: "head", layer: 7, head: 10 },
        title: "The circuit has backups",
        caption:
          "Knock out L7H10 alone and second-repeat loss barely moves (Δ −0.014 — the " +
          "others compensate). Zero all four induction heads together and Δ jumps to " +
          "+2.04, roughly 2.8× the sum of the single knockouts. Redundancy is why " +
          "single-head ablations understate circuits.",
      },
    ],
  },
  {
    id: "ioi",
    label: "How GPT-2 knows Mary",
    blurb: "The indirect-object circuit, recovered from scratch: attribution, intervention, occlusion.",
    model: "gpt2",
    steps: [
      {
        feature: "logit-attrib",
        trace: IOI,
        selection: { kind: "head", layer: 9, head: 6 },
        title: "The name-mover heads",
        caption:
          "“…John gave a drink to” → GPT-2 says “ Mary” at 44.6%. Decomposing that " +
          "margin exactly: heads L9H6 (+1.57) and L9H9 (+1.34) push “ Mary” hardest — " +
          "the name-mover heads of the IOI paper, recovered here from the raw forward.",
      },
      {
        feature: "causal-patching",
        title: "Proof by intervention",
        caption:
          "Attribution isn't causality — patching is. Swapping in the corrupted prompt " +
          "(“…Mary gave a drink to”) flips the answer logit-difference from +2.01 to " +
          "−1.79; copying single clean residual rows back in shows the name signal riding " +
          "the swapped-name position through L8, then handing off to the final position " +
          "at L9 — exactly where the name-movers write.",
      },
      {
        feature: "occlusion-vignette",
        trace: IOI,
        title: "Delete Mary, lose Mary",
        caption:
          "The bluntest test: occlude one token at a time and rerun. Delete the early " +
          "“ Mary” and the answer's log-prob drops 4.2 — the biggest hit of any content " +
          "token — and the prediction flips to “ John”. The model is really reading the " +
          "earlier name, not guessing from grammar.",
      },
    ],
  },
  {
    id: "sae-feature",
    label: "What an SAE feature is",
    blurb: "One dictionary direction, from dot on a map to firing pattern to its nearest twin.",
    model: "gpt2",
    steps: [
      {
        feature: "sae-decoder",
        selection: { kind: "saeFeature", id: 5856 },
        title: "A dot in the dictionary",
        caption:
          "Every dot is one decoder direction of the res-jb sparse autoencoder trained on " +
          "gpt2's layer-8 residual stream. Feature #5856 is pinned — position here says " +
          "little by itself. The next steps show what this direction does.",
      },
      {
        feature: "sae-piano-roll",
        trace: EIFFEL,
        selection: { kind: "saeFeature", id: 5856 },
        title: "Where it fires",
        caption:
          "Run the encoder on “The Eiffel Tower is located in the city of”: #5856 fires " +
          "on exactly one position — the final “ of”, activation 35.5 — and its top " +
          "vocabulary token is “ Calais”. A French-place feature, active precisely where " +
          "a city name must be predicted.",
      },
      {
        feature: "decoder-cosine-web",
        selection: { kind: "saeFeature", id: 4078 },
        title: "Features have twins",
        caption:
          "The dictionary isn't a clean basis: features #4078 and #11533 have decoder " +
          "cosine 1.00 — the SAE learned the same direction twice. The web shows every " +
          "feature's nearest neighbour; mutual pairs are drawn solid.",
      },
      {
        feature: "direction-compass",
        selection: { kind: "saeFeature", id: 5856 },
        title: "Directions, not neurons",
        caption:
          "The compass compares a feature's decoder direction against ALL 36,864 MLP " +
          "neurons and all 50k token embeddings. Most features align with no single " +
          "neuron — which is the point of the dictionary: concepts live in directions " +
          "that neurons only partially span.",
      },
    ],
  },
];

export function findTour(id: string): Tour | undefined {
  return TOURS.find((t) => t.id === id);
}

/** Add an episode to the registry (replacing one with the same id).
 *
 *  The three tours above are declared inline because they predate the registry;
 *  everything since is registered from `episodes.ts`, so that adding an episode
 *  touches exactly one file and nothing here has to import from the chrome it
 *  is rendered by. Replacement-by-id rather than append keeps a hot reload from
 *  stacking duplicates. */
export function register(tour: Tour): Tour {
  const at = TOURS.findIndex((t) => t.id === tour.id);
  if (at >= 0) TOURS[at] = tour;
  else TOURS.push(tour);
  return tour;
}

/** What the app can actually see, for `episodeAvailability` to check against.
 *  Passed in rather than read from module state so the rule is testable
 *  without a store, a network, or a built bundle. */
export interface EpisodeContext {
  /** dataset ids present in `out/index.json` */
  datasets: string[];
  /** channel ids for a dataset, or null when it has no `channels.json`
   *  (or when the sidecar has not been fetched yet) */
  channelsFor(datasetId: string): string[] | null;
  /** live Internals feature ids */
  features: string[];
  /** true once the channel sidecar for a dataset has been fetched — "still
   *  loading" and "measured and absent" must not render the same way */
  channelsLoaded?(datasetId: string): boolean;
  /** direction ids for a dataset, or null when it has no `directions.json`
   *  (or when that sidecar has not been fetched yet) */
  directionsFor?(datasetId: string): string[] | null;
  /** true once the direction sidecar for a dataset has been fetched */
  directionsLoaded?(datasetId: string): boolean;
}

export type EpisodeAvailability =
  | { state: "ready" }
  | { state: "pending"; reason: string }
  | { state: "unavailable"; reason: string };

/** May this episode be told, and if not, exactly what is missing.
 *
 *  Three outcomes, not two. "Pending" exists because a sidecar that has not
 *  arrived yet is not the same as one that does not exist, and telling the user
 *  an episode is unavailable while its data is in flight is its own small lie. */
export function episodeAvailability(tour: Tour, ctx: EpisodeContext): EpisodeAvailability {
  const m = tour.manifest;
  if (!m) return { state: "ready" };
  const hint = m.unavailable ? ` ${m.unavailable}` : "";

  if (m.dataset && !ctx.datasets.includes(m.dataset)) {
    return {
      state: "unavailable",
      reason: `needs the ${m.dataset} map, which this deploy does not ship.${hint}`,
    };
  }
  if (m.channels?.length) {
    const dsId = m.dataset ?? tour.model;
    const loaded = ctx.channelsLoaded?.(dsId) ?? true;
    const have = ctx.channelsFor(dsId);
    if (!loaded && have === null) {
      return { state: "pending", reason: `reading ${dsId}/channels.json…` };
    }
    const missing = m.channels.filter((c) => !(have ?? []).includes(c));
    if (missing.length) {
      return {
        state: "unavailable",
        reason:
          `needs ${missing.join(", ")} in ${dsId}/channels.json — ` +
          `run \`nebulai channels ${tour.model}\`.${hint}`,
      };
    }
  }
  if (m.directions?.length) {
    const dsId = m.dataset ?? tour.model;
    const loaded = ctx.directionsLoaded?.(dsId) ?? true;
    const have = ctx.directionsFor?.(dsId) ?? null;
    if (!loaded && have === null) {
      return { state: "pending", reason: `reading ${dsId}/directions.json…` };
    }
    const missing = m.directions.filter((d) => !(have ?? []).includes(d));
    if (missing.length) {
      return {
        state: "unavailable",
        reason: `needs ${missing.join(", ")} in ${dsId}/directions.json.${hint}`,
      };
    }
  }
  if (m.features?.length) {
    const missing = m.features.filter((f) => !ctx.features.includes(f));
    if (missing.length) {
      return {
        state: "unavailable",
        reason: `needs the ${missing.join(", ")} view, which is not live here.${hint}`,
      };
    }
  }
  return { state: "ready" };
}

/** Apply one tour step through the ordinary store actions. Selection rides the
 *  2a cross-view plumbing (InterpPage pushes it to the driver once ready), so
 *  a step behaves exactly like a user clicking the same entity.
 *
 *  This is the SYNCHRONOUS half: everything that is a plain store write. A step
 *  that changes model needs a fetch, so loading is `actions.runEpisodeStep`'s
 *  job and it calls this once the dataset is in place. */
export function applyTourStep(tour: Tour, stepIdx: number): void {
  const step = tour.steps[stepIdx];
  if (!step) return;
  const st = appStore.getState();
  if (step.feature) {
    st.setInterpFeature(step.feature);
    if (step.trace !== undefined) st.setInterpTrace(step.trace);
    st.setInterpSelection(step.selection ?? null);
  }
  // Applied after the feature switch: the driver publishes its own opening
  // selection while loading, and this must be the one that survives.
  if (step.steer) selectSteer(step.steer);
  if (step.page === "map") {
    // `channel: undefined` means "leave the lens as the previous step set it";
    // `channel: null` means "put it away". They are different instructions and
    // the distinction is what lets one episode narrate a lens across steps.
    if (step.channel !== undefined) st.setChannel(step.channel, step.channelWindow ?? null);
    else if (step.channelWindow !== undefined) st.setChannelWindow(step.channelWindow);
    if (step.search !== undefined) st.setMapQuery(step.search);
    if (step.mapSelection !== undefined) st.setSelection(step.mapSelection);
    // The axis is applied AFTER the selection because `setAxisDirection` does
    // not touch the selection and the reverse is also true — order is only
    // fixed so a reader of a step can predict the final state exactly.
    if (step.axis !== undefined) {
      st.setAxisDirection(step.axis);
      if (step.axis !== null) st.setAxisT(step.axisT ?? 1);
    } else if (step.axisT !== undefined) st.setAxisT(step.axisT);
  }
}
