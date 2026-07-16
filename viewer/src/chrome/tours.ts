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

import { appStore, type InterpSelection } from "../app/store";

export interface TourStep {
  /** registry feature id to show */
  feature: string;
  /** trace slug for per-trace features; omit for ownPrompts views */
  trace?: string;
  /** cross-view selection to pin (applied via setInterpSelection) */
  selection?: InterpSelection;
  title: string;
  caption: string;
}

export interface Tour {
  id: string;
  label: string;
  blurb: string;
  /** tours quote bundle-specific numbers — only offered on this model */
  model: string;
  steps: TourStep[];
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

/** Apply one tour step through the ordinary store actions. Selection rides the
 *  2a cross-view plumbing (InterpPage pushes it to the driver once ready), so
 *  a step behaves exactly like a user clicking the same entity. */
export function applyTourStep(tour: Tour, stepIdx: number): void {
  const step = tour.steps[stepIdx];
  if (!step) return;
  const st = appStore.getState();
  st.setInterpFeature(step.feature);
  if (step.trace !== undefined) st.setInterpTrace(step.trace);
  st.setInterpSelection(step.selection ?? null);
}
