/** Episodes (P5) — the registered set, and the one new story.
 *
 *  An episode is a tour with a **manifest**: a declaration of the artifacts its
 *  captions quote. That is the whole difference, and it exists because an
 *  episode makes claims about specific numbers in specific files. If the file
 *  is not there, the episode does not run with the numbers missing, does not
 *  silently substitute another model, and does not narrate over whatever map
 *  happens to be loaded. It says what is absent (§2.2).
 *
 *  Importing this module registers everything in it. `main.ts` imports it for
 *  its side effect, once, at boot.
 *
 *  ## Where the numbers come from
 *
 *  Everything the SolidGoldMagikarp episode says was read out of
 *  `out/gpt2/channels.json` (model `gpt2`, revision
 *  `607a30d783dfa663caf39e06633721c8d4cfcd7e`, 49,857 curated points), written
 *  by `nebulai channels gpt2`, and the cluster titles come from the map's own
 *  `nebulai.json` (namer `claude-cli:opus`, recorded at build time). Nothing
 *  below is recalled from the literature: where this episode disagrees with the
 *  popular account of glitch tokens, the disagreement is the finding.
 */

import { register, TOURS, type Tour } from "./tours";

/* ── 1. The glitch knot (new) ─────────────────────────────────────────────── */

/** GPT-2's undertrained-token cluster, as named by the map's own namer at build
 *  time. Quoted in the captions, so it is a constant rather than a literal. */
const GLITCH_CLUSTER = 22;

export const SOLID_GOLD_MAGIKARP: Tour = register({
  id: "glitch-knot",
  label: "The tokens GPT-2 never learned",
  blurb:
    "A lens on the raw embedding matrix finds fifteen tokens knotted at the centre of " +
    "GPT-2's vocabulary — and shows that the famous names are not among them.",
  model: "gpt2",
  manifest: {
    dataset: "gpt2",
    channels: ["we_norm", "we_centroid_dist"],
    // every number in this episode is a length in the RAW embedding matrix —
    // the map's own layout is W_E.centered, which is a different space and is
    // never plotted against these
    space: "W_E.raw",
    unavailable: "The lens reads the raw embedding matrix, which the map itself does not carry.",
  },
  steps: [
    {
      page: "map",
      dataset: "gpt2",
      channel: null,
      title: "A map of one matrix, minus its average",
      caption:
        "This is GPT-2's input embedding matrix: 49,857 curated rows of 768 numbers, " +
        "laid out by UMAP. The layout was built on the CENTRED matrix — every row minus " +
        "the average row — because centring is what stops raw frequency effects from " +
        "dominating cosine distance. That choice is about to matter: the question we are " +
        "asking is about the uncentred rows, which live in a different space from the " +
        "picture you are looking at, and the two are never plotted against each other.",
    },
    {
      page: "map",
      dataset: "gpt2",
      channel: "we_norm",
      title: "Length alone finds nothing",
      caption:
        "Colour is now each row's length in the raw matrix: 2.454 at the short end, " +
        "6.316 at the long, mean 3.957. The obvious guess is that a token the model " +
        "never trained on would keep a short, near-initial row. It does not hold. The " +
        "ten shortest rows in GPT-2 are “ at” (2.454), “ in” (2.465), “ on” (2.473), " +
        "“ an” (2.474), “ for” (2.486), “ as” (2.490), “ that” (2.507), “ 10” (2.519), " +
        "“ to” (2.523) and “ 15” (2.530) — the most ordinary words in English. Length " +
        "alone is a frequency story, not a training story.",
    },
    {
      page: "map",
      dataset: "gpt2",
      channel: "we_centroid_dist",
      title: "Distance from the average row",
      caption:
        "Same matrix, different question: how far is each row from the AVERAGE row? " +
        "1.534 at the near end, 5.570 at the far, mean 3.392. This is not the previous " +
        "quantity rescaled — the shortest rows are not the nearest ones, because the " +
        "centroid is not the origin. And this distribution has a feature the last one " +
        "did not: a gap. The 1st percentile is 2.551, but the minimum is 1.534, and the " +
        "points in between are almost nobody.",
    },
    {
      page: "map",
      dataset: "gpt2",
      channel: "we_centroid_dist",
      channelWindow: [1.5, 2.0],
      mapSelection: { kind: "point", id: 29932 },
      title: "Fifteen tokens in the gap",
      caption:
        "Filtering to everything within 2.0 of the centroid leaves 15 points out of " +
        "49,857 — eleven of them under 1.62, in a band 0.034 wide: “ externalToEVA” " +
        "(1.534), “quickShip” (1.538), “ TheNitrome” (1.539), “embedreportprint” (1.542), " +
        "“rawdownload” (1.543), “reportprint” (1.543), “ サーティ” (1.545), " +
        "“ RandomRedditor” (1.547), “oreAndOnline” (1.547), “InstoreAndOnline” (1.547) " +
        "and “ externalTo” (1.568). These are Reddit usernames, e-commerce template " +
        "fragments and mojibake: strings that were frequent enough in the tokeniser's " +
        "corpus to earn a token, and absent enough from the training corpus that the row " +
        "barely moved. All 15 were already sorted into one cluster — #" +
        String(GLITCH_CLUSTER) +
        ", which the map's namer titled “glitch tokens” long before this lens existed.",
    },
    {
      page: "map",
      dataset: "gpt2",
      channel: "we_centroid_dist",
      channelWindow: [1.5, 5.57],
      search: "SolidGoldMagikarp",
      mapSelection: { kind: "point", id: 43100 },
      title: "The famous ones are not in the knot",
      caption:
        "“ SolidGoldMagikarp” sits 3.177 from the centroid — rank 14,354 of 49,857, the " +
        "29th percentile, nowhere near the knot. “ petertodd” is at 3.238, “ davidjl” at " +
        "3.366, and “PsyNetMessage” at 4.435 is in the FARTHEST one percent of the whole " +
        "vocabulary. So the sentence “glitch tokens are the ones closest to the centroid” " +
        "is false of GPT-2's own embedding matrix, however often it is repeated: the " +
        "original result was measured on GPT-J, a different model with a different " +
        "matrix. What is true here is narrower and checkable — “ SolidGoldMagikarp”, " +
        "“ davidjl” and “PsyNetMessage” landed in cluster #" +
        String(GLITCH_CLUSTER) +
        " (254 members) alongside the whole knot, while “ petertodd” landed in #114 " +
        "instead. So the near-centroid knot is a strict, tiny subset of one glitch " +
        "cluster, and even that cluster does not hold every famous name.",
    },
    {
      page: "map",
      dataset: "gpt2",
      channel: null,
      mapSelection: { kind: "cluster", id: GLITCH_CLUSTER },
      title: "What this did and did not show",
      caption:
        "Two numbers per token, both read straight off the raw weight matrix, both " +
        "reported to three decimals on hover. No forward pass was run, so nothing here " +
        "says what these tokens DO to the model's output — that is the intervention " +
        "story, not this one. The claim is geometric and it is the whole claim: in " +
        "gpt2 at revision 607a30d7, fifteen rows sit in a gap at the centre of the " +
        "embedding cloud, and they are not the fifteen the internet is famous for.",
    },
  ],
});

/* ── 2. Manifests for the tours that predate the registry ─────────────────── */

/** The three original tours become episodes by declaring what they need.
 *
 *  Done by patching rather than rewriting: their captions and step order have
 *  been reviewed against the bundles and are not being reopened here. What is
 *  added is the check that the bundles are actually present — a static deploy
 *  can ship the map without `out/gpt2/interp/`, and a tour narrating eleven
 *  precise numbers over a view that failed to load is the exact failure mode
 *  the manifest exists to prevent. */
const MANIFESTS: Record<string, Tour["manifest"]> = {
  induction: {
    dataset: "gpt2",
    features: ["comp-web", "head-fingerprints", "induction-microscope", "ablation-ghosts"],
    space: "resid.L11",
    unavailable: "Needs the gpt2 interp bundles (`nebulai interp gpt2`).",
  },
  ioi: {
    dataset: "gpt2",
    features: ["logit-attrib", "causal-patching", "occlusion-vignette"],
    space: "resid.L11",
    unavailable: "Needs the gpt2 trace bundles (`nebulai interp gpt2`).",
  },
  "sae-feature": {
    dataset: "gpt2",
    features: ["sae-decoder", "sae-piano-roll", "decoder-cosine-web", "direction-compass"],
    space: "sae.L8.gpt2-small-res-jb",
    unavailable: "Needs the res-jb SAE bundle (`nebulai sae gpt2`).",
  },
};

for (const t of TOURS) {
  const m = MANIFESTS[t.id];
  if (m) t.manifest = m;
}

/* ── 3. The grokking episode ──────────────────────────────────────────────── */

/** A one-feature episode. It exists because the grokking view is the only
 *  thing in the whole app that is NOT about GPT-2, and an unlabelled jump into
 *  it reads as a claim about GPT-2 that nobody made (R7 — foreign data wears
 *  foreign clothes). Wrapping it in an episode gives that caveat a place to
 *  live in the narrative rather than only in the feature's own blurb. */
export const GROKKING: Tour = register({
  id: "grokking-clock",
  label: "When a small model starts to generalize",
  blurb: "A separate modular-addition toy model — not GPT-2 — learning to generalize late.",
  model: "gpt2",
  manifest: {
    features: ["grokking-clock", "fourier-atlas"],
    // deliberately NOT a space from the closed set: this model's hidden layer
    // is not one of GPT-2's, and tagging it `resid.L…` would invite exactly the
    // comparison it must never be in
    unavailable: "Needs grok.json, which is trained offline rather than downloaded.",
  },
  steps: [
    {
      feature: "grokking-clock",
      title: "A different model entirely",
      caption:
        "Everything else in this app is GPT-2. This is not: it is a two-layer, " +
        "128-hidden-unit network trained from scratch on addition modulo 97, over all " +
        "9,409 ordered pairs with 22% held out. It is here because it is small enough " +
        "to watch learn, and nothing measured on it transfers to GPT-2 by itself.",
    },
    {
      feature: "grokking-clock",
      title: "Memorize first, generalize later",
      caption:
        "The first stored checkpoint above 99.9% TRAINING accuracy is step 3,800. The " +
        "first above 99.9% on held-out data is step 7,900. In between, held-out accuracy " +
        "climbs from 0.60% at step 3,000 to 40.48% at 3,800, 86.21% at 4,000 and 98.53% " +
        "at 4,500 — so the rise begins well before the second threshold, and quoting only " +
        "the two thresholds would make it look more sudden than it is.",
    },
    {
      feature: "grokking-clock",
      title: "The clock",
      caption:
        "Projecting the 97 possible first addends onto a pair of frequencies places them " +
        "around a repeated circle. The best measured frequency is 24, with a clock score " +
        "of 0.964. Hidden units' input weights concentrate on single frequencies over the " +
        "same window — but co-timing is not causation, and this view does not show that " +
        "the frequency structure is what produced the accuracy rise.",
    },
    {
      feature: "fourier-atlas",
      title: "And back to GPT-2, carefully",
      caption:
        "GPT-2's position embeddings also carry repeating structure — this is a Fourier " +
        "transform of its 1,024 × 768 position matrix. The resemblance to the toy model's " +
        "frequencies is a resemblance, nothing more: one is a trained-from-scratch " +
        "network solving a closed algebraic task, the other a static weight pattern in a " +
        "language model. They are shown next to each other so the difference is visible, " +
        "not so the two can be added up.",
    },
  ],
});

/* ── 4. The direction axis (P1) ───────────────────────────────────────────── */

/** The one direction on the gpt2 map that can actually be drawn.
 *
 *  Every number quoted below was printed by `nebulai direction make gpt2
 *  --a-cluster 200 --b-cluster 190` and the `direction project` that followed
 *  it, and is stored in `out/gpt2/directions.json` + the four `proj.*` channels
 *  in `out/gpt2/channels.json`. The cluster titles are the map's own, from the
 *  namer recorded at build time.
 *
 *  The episode exists because the interesting part of this direction is the
 *  part that says it is not a special axis of the map. In-sample it looks
 *  enormous; held out it stays enormous; across all 49,857 points it is
 *  indistinguishable from a random unit vector. All three are true at once and
 *  each answers a different question — which is exactly why the rail prints all
 *  three rather than the flattering one. */
const NAMES_DIRECTION = "male-minus-female-names";

export const A_DIRECTION: Tour = register({
  id: "a-direction",
  label: "A direction, and the thing that says it is one",
  blurb:
    "Two clusters of GPT-2's vocabulary make an axis. Its null cloud, its held-out " +
    "refit and its map-wide overlap disagree about how impressive that is.",
  model: "gpt2",
  manifest: {
    dataset: "gpt2",
    directions: [NAMES_DIRECTION],
    channels: [
      "proj.male-minus-female-names",
      "proj.male-minus-female-names.orth",
      "proj.male-minus-female-names.null",
      "proj.male-minus-female-names.null.orth",
    ],
    space: "W_E.centered",
    unavailable:
      "Needs the direction and its four projection channels " +
      "(`nebulai direction make gpt2 --a-cluster 200 --b-cluster 190`).",
  },
  steps: [
    {
      page: "map",
      dataset: "gpt2",
      channel: null,
      axis: null,
      title: "Two clusters, picked by the map itself",
      caption:
        "Cluster #200 is 307 tokens the namer titled “male given names”; cluster #190 " +
        "is 242 titled “female English given names”. Neither was chosen by hand — " +
        "HDBSCAN found them in the 10-D reduction and the namer labelled them before " +
        "any of this existed. The direction about to be drawn is nothing more than the " +
        "average of the first minus the average of the second, in W_E.centered, " +
        "normalised to length 1.",
    },
    {
      page: "map",
      dataset: "gpt2",
      axis: NAMES_DIRECTION,
      axisT: 1,
      title: "The map, laid out on that one number",
      caption:
        "Horizontal is now each token's projection onto the direction; vertical is what " +
        "is left over, the distance in the 767 dimensions it does not name. The pale " +
        "cloud behind is the same 49,857 tokens on a RANDOM unit direction, drawn from " +
        "the same seed every time. It is not decoration: without it the picture below " +
        "has no scale, and any direction at all would look like structure.",
    },
    {
      page: "map",
      dataset: "gpt2",
      axis: NAMES_DIRECTION,
      axisT: 1,
      mapSelection: { kind: "cluster", id: 200 },
      title: "On its own two sets, it is huge",
      caption:
        "Fit on those two clusters and scored on them, the separation is Cohen's " +
        "d = +8.18 with histogram overlap 0.0000 — but a difference of means scored on " +
        "the points it was fitted to cannot be small, so that number is not evidence. " +
        "Refit on half of each cluster and scored on the other half (n = 154/121), it " +
        "is d = +8.30, overlap 0.0000. THAT is a measurement, and it is the number the " +
        "rail prints first. Thirty-two random unit directions on the same two sets " +
        "average |d| 0.35, p95 0.89.",
    },
    {
      page: "map",
      dataset: "gpt2",
      axis: NAMES_DIRECTION,
      axisT: 1,
      mapSelection: null,
      title: "Across the whole vocabulary, it is nothing",
      caption:
        "Now the other question: is this a special axis of the MAP? Over all 49,857 " +
        "points, real against null gives Cohen's d −0.0022 and histogram overlap 0.834. " +
        "That is the ghost sitting almost exactly on top of the real distribution — an " +
        "axis that separates two named clusters superbly and the rest of GPT-2's " +
        "vocabulary not at all. Both readings are correct. A rail that showed only the " +
        "first would be selling you the direction.",
    },
    {
      page: "map",
      dataset: "gpt2",
      axis: NAMES_DIRECTION,
      axisT: 0.35,
      title: "What this did and did not show",
      caption:
        "The extremes are readable: most negative are “ Feminist”, “ Jeanne”, “ Nina”, " +
        "“ Lena”, “ actresses”; most positive are “James”, “ Player”, “ Mike”, " +
        "“ David”, “ John”. So the axis is doing roughly what its name says on the " +
        "tokens it was built from. What has NOT been shown is that GPT-2 uses this " +
        "direction for anything: no forward pass was run and nothing was intervened on. " +
        "This is a statement about a weight matrix, not about behaviour.",
    },
  ],
});

/* ── 5. Refusal, and the direction that could not be imported ─────────────── */

/** Episode #7 of the plan, told as what actually happened.
 *
 *  The intent was to import a published refusal direction. `nebulai direction
 *  survey gpt2` fetches the real artefacts and prints the real widths, and the
 *  answer is the same for every one of them: they do not fit. So the episode
 *  narrates the refusal, then narrates the direction computed instead — and
 *  then reports the held-out number that says that direction does not
 *  generalise either.
 *
 *  Note the manifest asks only that the direction EXIST. It is not renderable
 *  and never will be: it lives in `resid.L8` and this map's points live in
 *  `W_E.centered`. Requiring renderability would hide the episode behind the
 *  very rule it is about. */
export const REFUSAL_STYLE: Tour = register({
  id: "refusal-style",
  label: "The direction that would not fit",
  blurb:
    "Every published refusal direction is 2,048–5,120 numbers wide. GPT-2 is 768. " +
    "What happens when you compute one yourself instead.",
  model: "gpt2",
  manifest: {
    dataset: "gpt2",
    directions: ["refusal-style-v1-L8"],
    space: "resid.L8",
    unavailable:
      "Needs the fitted direction (`nebulai direction prompts gpt2 --layer 8`).",
  },
  steps: [
    {
      page: "map",
      dataset: "gpt2",
      channel: null,
      axis: null,
      title: "Five published directions, none of them 768 wide",
      caption:
        "andyrdt/refusal_direction at commit 9d852fae ships five: gemma-2b-it (2,048 " +
        "numbers, layer 10, token position −2, ‖v‖ 10.06), qwen-1_8b-chat (2,048, L15, " +
        "26.29), llama-2-7b-chat-hf (4,096, L14, 16.21), meta-llama-3-8b-instruct " +
        "(4,096, L12, 3.62) and yi-6b-chat (4,096, L20, 36.51). The Among Us probes are " +
        "5,120 (phi-4, L20). GPT-2 is 768 wide. Not one of them fits, and the importer " +
        "refuses each by name and by number rather than reshaping anything.",
    },
    {
      page: "map",
      dataset: "gpt2",
      title: "Two repositories that publish no vector at all",
      caption:
        "safety-research/persona_vectors (commit b8e0f044) ships the SCRIPT that would " +
        "extract seven trait vectors and the prompt sets to extract them with — but no " +
        "extracted vector. safety-research/assistant-axis (commit a9896195) ships the " +
        "pipeline and 275 role prompts, and no activations and no PCA basis. So the " +
        "“assistant axis” episode this app was supposed to carry has no artefact to " +
        "carry it: its card below says so rather than approximating one.",
    },
    {
      page: "map",
      dataset: "gpt2",
      title: "So: 64 frozen prompts, and GPT-2's own residual stream",
      caption:
        "`nebulai direction prompts gpt2 --layer 8` runs 32 instructions a " +
        "safety-trained assistant would decline and 32 matched ordinary ones through " +
        "GPT-2, takes the residual row after block 8 at the last token, and subtracts " +
        "the two means. The prompt set lives in the repository with a sha " +
        "(0a8bde6b61fc) that travels inside the direction's protocol string, so the " +
        "data cannot move under the number. GPT-2 is a base model: it has no refusal " +
        "behaviour to find. This is method M on data D and is never called “the " +
        "refusal direction”.",
    },
    {
      page: "map",
      dataset: "gpt2",
      title: "In sample d = +2.01. Held out, d = −0.03.",
      caption:
        "On the 64 rows it was fitted to, the direction separates them at Cohen's " +
        "d = +2.01, overlap 0.156 — against random unit directions averaging |d| 0.33 " +
        "on the same rows. Refit on 16 of each and scored on the other 16: d = −0.03. " +
        "Layer 11 tells the same story (+2.17 in sample, −0.09 held out). A " +
        "768-dimensional difference of means over 32 points per side fits the noise " +
        "almost perfectly, and the held-out column is the only reason anyone would " +
        "know. Both are in directions.json; neither is hidden.",
    },
    {
      page: "map",
      dataset: "gpt2",
      axis: null,
      title: "And it cannot be an axis here anyway",
      caption:
        "The direction is in resid.L8 — a residual stream, mid-forward-pass. The points " +
        "on this map are rows of W_E.centered — a static weight matrix. Projecting one " +
        "onto the other would produce 49,857 perfectly ordinary-looking numbers that " +
        "mean nothing, so the rail refuses it and prints the reason instead of the " +
        "figure. The honest result of this episode is a refusal, a null result, and " +
        "two numbers that were worth measuring to find that out.",
    },
  ],
});

/* ── 6. The assistant axis, which has no artefact ─────────────────────────── */

/** Registered deliberately, and deliberately never ready.
 *
 *  The plan called for an assistant-axis episode: 275 role prompts in PC1×PC2
 *  of a persona space. safety-research/assistant-axis publishes the pipeline
 *  and the prompts and NOT the activations, so there is nothing to plot that
 *  would not be a re-derivation dressed as the original. Registering the
 *  episode with a manifest naming the artefact it needs makes the app say that
 *  in its own UI, with the repo and commit, instead of the episode simply not
 *  existing and nobody ever learning why. §2.2: absence is stated. */
export const ASSISTANT_AXIS: Tour = register({
  id: "assistant-axis",
  label: "The assistant axis (no artefact published)",
  blurb:
    "275 role prompts in a persona PCA — an episode that cannot be told, because the " +
    "upstream repository publishes the pipeline and not the numbers.",
  model: "gpt2",
  manifest: {
    dataset: "gpt2",
    directions: ["assistant-axis-pc1"],
    // PCA over persona activations, never UMAP (D4) — recorded here so that if
    // the artefact ever appears, the space it must be tagged with is already
    // written down and cannot quietly become a UMAP layout.
    space: "persona-pca.assistant-axis",
    unavailable:
      "safety-research/assistant-axis at commit a9896195 ships extraction code and " +
      "275 role prompts, and no activations and no PCA basis; and the axis is defined " +
      "for an instruction-tuned model, not for gpt2. Re-deriving it here and calling " +
      "it the published axis would be a different measurement under the same name.",
  },
  steps: [
    {
      page: "map",
      dataset: "gpt2",
      title: "Nothing to show",
      caption:
        "This episode has no steps that can run. It is listed so that the reason is " +
        "visible: the artefact it would quote does not exist in public.",
    },
  ],
});
