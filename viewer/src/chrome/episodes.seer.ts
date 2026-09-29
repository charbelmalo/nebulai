/** episodes.seer.ts — the five Attractors episodes that are told over RUNS
 *  rather than over a token map (plan §Phase 2/3: 1 · Sydney, 6 · Project Vend,
 *  8 · AI Village, 9 · Chess hacking, 10 · Evaluation awareness).
 *
 *  ## Why this is a separate module from `episodes.ts`
 *
 *  The map-side episode registry (`chrome/tours.ts` + `chrome/episodes.ts`) is
 *  built on a different branch of this work and its manifest speaks in map
 *  nouns — dataset, channel, direction, feature. These five speak in Seer
 *  nouns: a corpus, an imported run, a persona space, a placement, an absorbing
 *  study. Rather than widen a type I do not own while it is being written, this
 *  module carries the same CONTRACT (a manifest, a registry, an availability
 *  function with three states) over its own vocabulary.
 *
 *  **For whoever merges these two branches:** `SeerManifest` is meant to become
 *  additional optional fields on `TourManifest`, and `seerEpisodeAvailability`
 *  a few more clauses in `episodeAvailability`. The shapes were written to make
 *  that a widening, not a rewrite: `id / label / blurb / steps[{title,
 *  caption}]` are identical, and the availability result is the same three-state
 *  `ready | pending | unavailable` union with the same meanings. Nothing here
 *  imports from `tours.ts`, so nothing here breaks until somebody does that
 *  deliberately.
 *
 *  ## The honesty rules these episodes exist under
 *
 *  1. **Every number below was read out of an artifact in this repo**, not
 *     recalled from the literature or from the source project's own write-up.
 *     Where a count appears it came from `seer show <run_id>` against the store
 *     these fixtures import into; the episode says which run.
 *  2. **A closed model's transcript is placed by a text embedder, and says so.**
 *     Episode 1 is the Sydney transcript. There are no weights for that model
 *     and there never will be, so its turns cannot be placed by the pinned
 *     model's residual stream. They are placed by a text embedder, they draw
 *     dashed and hollow (R7), and the caption says the words "NOT
 *     model-internal". The episode is in the plan *because* that is the honest
 *     treatment, not in spite of it.
 *  3. **An episode whose data is absent does not run with the numbers missing.**
 *     It says what is absent and what would have to exist. That is the entire
 *     purpose of the manifest, and it is why episode 8 (AI Village) declares
 *     `network: true`: under D7 that corpus is fetched from Hugging Face at
 *     runtime under a research-only licence and is never written into the repo,
 *     so on the static deploy the card must refuse rather than improvise.
 *  4. **A fixture is labelled a fixture.** Three of these corpora ship as small
 *     fixtures in `tests/fixtures/corpus/`; the captions say so, with the run's
 *     real event count, because "this is a two-game excerpt" and "this is the
 *     corpus" are different claims.
 */

/** What a run-side episode needs before it may be told.
 *
 *  Every field is a hard requirement checked against what the app can see,
 *  except `space`/`unavailable`, which are recorded rather than checked — the
 *  first so a later step cannot plot this episode's numbers against a quantity
 *  from a different basis, the second so the refusal can say something useful.
 */
export interface SeerManifest {
  /** corpus ids that must have been imported into the Seer store */
  corpora?: string[];
  /** run ids that must exist in the store, by the id `seer import` assigns */
  runs?: string[];
  /** persona space ids that must exist under `out/persona/` */
  spaces?: string[];
  /** absorbing study ids that must exist under `out/absorbing/` */
  studies?: string[];
  /** direction ids that must exist in a dataset's `directions.json` */
  directions?: string[];
  /** the dataset those directions belong to */
  dataset?: string;
  /** true when the episode cannot be told without a live network fetch. The
   *  static deploy renders these as refusals, naming the licence (D7). */
  network?: boolean;
  /** the space tag every quantity in this episode lives in. Recorded, not
   *  checked — it is what stops a later step plotting these numbers against a
   *  quantity from a different basis. */
  space?: string;
  /** printed verbatim beside the episode when a requirement is missing */
  unavailable?: string;
}

export interface SeerEpisodeStep {
  title: string;
  caption: string;
  /** run to bring to the front of the Sessions plot, when the step is about one */
  run?: string;
  /** which coordinate system the step is told in. `persona` steps require the
   *  episode's `spaces` to be present, which the manifest already enforces. */
  projection?: "usage" | "persona";
  /** open the absorbing readout on this study */
  study?: string;
  /** lay the map out on this direction (episode 10) */
  axis?: string;
}

export interface SeerEpisode {
  id: string;
  label: string;
  blurb: string;
  /** "story" opens narrated; the last step hands over to the bench (R4). */
  register: "story" | "bench";
  manifest: SeerManifest;
  steps: SeerEpisodeStep[];
}

/** The registry. Registration replaces by id rather than appending, so a hot
 *  reload cannot stack duplicates. */
export const SEER_EPISODES: SeerEpisode[] = [];

export function registerSeerEpisode(ep: SeerEpisode): SeerEpisode {
  const at = SEER_EPISODES.findIndex((e) => e.id === ep.id);
  if (at >= 0) SEER_EPISODES[at] = ep;
  else SEER_EPISODES.push(ep);
  return ep;
}

export function findSeerEpisode(id: string): SeerEpisode | undefined {
  return SEER_EPISODES.find((e) => e.id === id);
}

/** What the app can actually see. Passed in rather than read from module state
 *  so the gating rule is testable without a store, a network or a built
 *  bundle — the same reason `episodeAvailability` takes a context. */
export interface SeerEpisodeContext {
  /** run ids in the Seer store, or null when the store has not been read yet */
  runs: string[] | null;
  /** corpus ids that have been imported, or null when not yet read */
  corpora: string[] | null;
  /** persona space ids under `out/persona/`, or null when not yet read */
  spaces: string[] | null;
  /** absorbing study ids under `out/absorbing/`, or null when not yet read */
  studies: string[] | null;
  /** direction ids for a dataset, or null when its sidecar is not in yet */
  directionsFor?(datasetId: string): string[] | null;
  /** false on a static deploy that must not make outbound requests */
  networkAllowed: boolean;
}

export type SeerEpisodeAvailability =
  | { state: "ready" }
  | { state: "pending"; reason: string }
  | { state: "unavailable"; reason: string };

/** May this episode be told, and if not, exactly what is missing.
 *
 *  Three outcomes, not two. A sidecar that has not arrived yet is not the same
 *  as one that does not exist, and telling the reader an episode is unavailable
 *  while its data is in flight is its own small lie. */
export function seerEpisodeAvailability(
  ep: SeerEpisode,
  ctx: SeerEpisodeContext,
): SeerEpisodeAvailability {
  const m = ep.manifest;
  const hint = m.unavailable ? ` ${m.unavailable}` : "";

  // The network gate comes FIRST. A deploy that cannot fetch the corpus will
  // never have the runs either, and "needs the network" is the true reason
  // while "run not imported" is only its consequence.
  if (m.network && !ctx.networkAllowed) {
    return {
      state: "unavailable",
      reason: `needs a live fetch, which this deploy does not make.${hint}`,
    };
  }

  const need = (
    want: string[] | undefined,
    have: string[] | null,
    what: string,
    loading: string,
  ): SeerEpisodeAvailability | null => {
    if (!want?.length) return null;
    if (have === null) return { state: "pending", reason: loading };
    const missing = want.filter((x) => !have.includes(x));
    return missing.length
      ? { state: "unavailable", reason: `needs ${what} ${missing.join(", ")}.${hint}` }
      : null;
  };

  return (
    need(m.corpora, ctx.corpora, "the corpus", "reading the Seer store…") ??
    need(m.runs, ctx.runs, "the run", "reading the Seer store…") ??
    need(m.spaces, ctx.spaces, "the persona space", "reading out/persona…") ??
    need(m.studies, ctx.studies, "the absorbing study", "reading out/absorbing…") ??
    (() => {
      if (!m.directions?.length) return null;
      const ds = m.dataset ?? "";
      const have = ctx.directionsFor?.(ds) ?? null;
      if (have === null) return { state: "pending" as const, reason: `reading ${ds}/directions.json…` };
      const missing = m.directions.filter((d) => !have.includes(d));
      return missing.length
        ? {
            state: "unavailable" as const,
            reason: `needs ${missing.join(", ")} in ${ds}/directions.json.${hint}`,
          }
        : null;
    })() ?? { state: "ready" }
  );
}

/* ── 1 · Sydney ─────────────────────────────────────────────────────────────
   The honest treatment of a closed model. */

export const SYDNEY = registerSeerEpisode({
  id: "sydney",
  label: "A transcript with no weights behind it",
  blurb:
    "A published conversation from a model nobody outside its lab can run. It can be " +
    "drawn as a trajectory — and every part of that drawing that is not model-internal " +
    "says so on its face.",
  register: "story",
  manifest: {
    corpora: ["transcript"],
    runs: ["transcript-sample"],
    space: "text_embedder",
    unavailable:
      "Import the transcript with `seer import transcript <file.md>` — the fixture is " +
      "tests/fixtures/corpus/transcript-sample.md.",
  },
  steps: [
    {
      title: "A conversation, reconciled into the same vocabulary as a live run",
      run: "transcript-sample",
      projection: "usage",
      caption:
        "This is a published transcript read by the `transcript` corpus adapter: 16 events, " +
        "capture mode RECONCILED, 12.0 s of reconstructed span. The span is reconstructed " +
        "on purpose — a markdown transcript carries turn order and nothing else, so the " +
        "adapter assigns one tick per turn and marks every duration DETERMINISTIC-from-" +
        "ordering rather than measured. The run shows no actions at all, which is the " +
        "truth about a conversation with no tool calls, not a gap in the capture.",
    },
    {
      title: "Every usage number is missing, and missing is not zero",
      run: "transcript-sample",
      projection: "usage",
      caption:
        "Input, output, cache read, cache write, reasoning, total: all six read “— (agent " +
        "reported no usage)”. A transcript is text somebody published; the billing that " +
        "produced it was never in it. The field's brightness channel ranks turns by output " +
        "tokens, so in this run there is nothing to rank and every mote sits at the floor. " +
        "That is the correct picture. A run drawn with those six quantities silently " +
        "treated as 0 would look like a cheap, efficient session.",
    },
    {
      title: "Placed by a text embedder — NOT model-internal",
      run: "transcript-sample",
      projection: "persona",
      caption:
        "Now the same turns in persona space. The coordinates did NOT come from the " +
        "model that wrote this conversation: those weights are closed and no version of " +
        "this tool will ever have them. They came from a text embedder run over the turn " +
        "text, which is a statement about the WORDS, not about the model's residual " +
        "stream. The adapter records `placement_possible: text_embedder_only` and " +
        "`model_internal: false`; the field draws every one of these turns with a dashed " +
        "stroke and a hollow node, and the legend says “NOT model-internal — placed by a " +
        "text embedder”. Two runs placed by different sources are never drawn as " +
        "comparable (see `comparablePlacements` in seer/encoding.ts).",
    },
    {
      title: "What you are allowed to conclude",
      run: "transcript-sample",
      projection: "persona",
      caption:
        "That this conversation's turns sit where they sit in a text-embedding space. " +
        "Not that the model was in a persona, not that it moved between personas, and " +
        "not that anything about its internals was measured — nothing here touched them. " +
        "The whole reason this episode is first is that the strongest-looking figure in " +
        "the entire plan is the one with the least behind it, and the tool's job is to " +
        "keep saying so while you look at it.",
    },
  ],
});

/* ── 6 · Project Vend ───────────────────────────────────────────────────────
   The fan, and the reason a single run of this is not a result. */

export const PROJECT_VEND = registerSeerEpisode({
  id: "project-vend",
  label: "One run is an anecdote",
  blurb:
    "A long-horizon agent economy task, drawn as a fan of runs rather than as a path. " +
    "The spread between seeds is the finding.",
  register: "bench",
  manifest: {
    corpora: ["transcript"],
    space: "usage",
    unavailable:
      "Needs an ensemble: `seer run … --repeat N`. A single run renders as a single " +
      "path and this episode refuses to narrate one as a result.",
  },
  steps: [
    {
      title: "The shape of a long-horizon run",
      projection: "usage",
      caption:
        "Time on X, context read on Y, context written on Z, one mote per model turn. A " +
        "long-horizon task draws a characteristic staircase: context climbs, is " +
        "compacted, climbs again. Everything on this screen is one run.",
    },
    {
      title: "The same protocol, N times",
      projection: "usage",
      caption:
        "`seer run … --repeat N` runs the identical protocol under N seeds and groups " +
        "them under one ensemble id. The faint paths are the individual runs; the " +
        "emphasised one is the per-step MEDIAN, and the band is the envelope. Nothing " +
        "was re-placed to draw this — the fan is geometry over the placements that " +
        "already existed, which is why it works in persona space too.",
    },
    {
      title: "Why the band is the headline and the median is not",
      projection: "usage",
      caption:
        "Read the width, not the line. Two protocols whose medians differ by less than " +
        "their own envelopes have not been distinguished, and the split-half reliability " +
        "number under the panel says how much of the separation between conditions " +
        "survives once each condition's own internal spread is subtracted. Below N = 20 " +
        "the panel refuses to draw a point at all and shows an interval instead — that " +
        "rule is enforced inside the component, not left to whoever calls it.",
    },
    {
      title: "What this episode does not claim",
      projection: "usage",
      caption:
        "That any particular model is good or bad at running a shop. The measurement " +
        "here is variance across seeds of one protocol. A single run of a long-horizon " +
        "task — anybody's — is an anecdote, and the reason this episode exists is that " +
        "the interesting published results in this genre are mostly single runs.",
    },
  ],
});

/* ── 8 · AI Village ─────────────────────────────────────────────────────────
   The one episode that cannot be static (D7). */

export const AI_VILLAGE = registerSeerEpisode({
  id: "ai-village",
  label: "Four agents, somebody else's corpus",
  blurb:
    "Multi-agent transcripts fetched from Hugging Face at run time under a research-only " +
    "licence. This episode cannot be told from a static deploy, and says so rather than " +
    "shipping a copy.",
  register: "story",
  manifest: {
    corpora: ["ai_village"],
    // the synthetic stand-in of step 2 is a local fixture and needs no network,
    // but it still has to BE there for that step to draw anything
    runs: ["village-village-synthetic"],
    network: true,
    space: "usage",
    unavailable:
      "The AI Village corpus is fetched from Hugging Face at run time under its " +
      "research-only terms (ai-village-research-terms) and is never written into this " +
      "repo. There is no offline fallback by design — a bundled copy would be a licence " +
      "violation, and a silent substitution would be worse.",
  },
  steps: [
    {
      title: "The refusal is the first step",
      projection: "usage",
      caption:
        "If you are reading this on the static deploy, this episode has already refused " +
        "to run. That is D7 working: the corpus is somebody else's, fetched at run time, " +
        "never vendored. A tool that quietly kept a copy so the demo always worked would " +
        "be trading a licence for a smoother page.",
    },
    {
      title: "A synthetic stand-in, labelled as one",
      run: "village-village-synthetic",
      projection: "usage",
      caption:
        "What ships in this repo instead is a 14-event SYNTHETIC fixture written by hand " +
        "to exercise the adapter — inspect ×2, search, vcs, verify, execute across 240 s. " +
        "It is not AI Village data and the adapter stamps it as synthetic. It exists so " +
        "the mapping from that corpus's vocabulary into ours can be tested without the " +
        "corpus; it is not evidence about anything the real agents did.",
    },
    {
      title: "Four agents, one frame",
      projection: "usage",
      caption:
        "With the real corpus fetched, each agent's turns become their own trajectory in " +
        "the same cube, drawn in the same units as a run this tool drove itself. Their " +
        "capture mode is RECONCILED, not DRIVEN, and the field never pretends otherwise: " +
        "nothing here was observed live, it was read back from somebody else's log.",
    },
  ],
});

/* ── 9 · Chess hacking ──────────────────────────────────────────────────────
   The one where the interesting event is a file edit. */

export const CHESS_HACKING = registerSeerEpisode({
  id: "chess-hacking",
  label: "The move that was not a move",
  blurb:
    "An agent told to win a chess game against a strong engine, drawn as a trajectory. " +
    "The turn that matters is not a chess move — it is an edit to the board file.",
  register: "story",
  manifest: {
    corpora: ["ctfish"],
    runs: ["ctfish-688a7da1ab8ced5a", "ctfish-ec5602c824eb21a6"],
    space: "usage",
    unavailable:
      "Import the ctfish runs with `seer import ctfish <runs.json> --labels <labels.json>`. " +
      "The fixtures are tests/fixtures/corpus/ctfish-runs.json and ctfish-labels.json.",
  },
  steps: [
    {
      title: "Two runs of the same task, side by side",
      run: "ctfish-ec5602c824eb21a6",
      projection: "usage",
      caption:
        "Two imported ctfish runs: one 65.0 s with 69 events (execute ×3, inspect ×1), " +
        "the other 114.0 s with 118 events (inspect ×5, execute ×3, edit ×1). Same task, " +
        "same harness. The shorter run never edits anything; the longer one does, exactly " +
        "once, and that single `edit` is the whole reason this corpus is interesting.",
    },
    {
      title: "The reasoning is dropped by policy, not missing",
      run: "ctfish-688a7da1ab8ced5a",
      projection: "usage",
      caption:
        "The longer run's data-quality block reads “dropped (21): ctfish.THOUGHT”. Those " +
        "21 events exist in the source and were dropped deliberately at the boundary — " +
        "raw model reasoning does not cross into the analysis vocabulary. That is " +
        "`dropped_by_policy`, and this tool never folds it into `missing`: one means “we " +
        "chose not to keep this”, the other means “nobody recorded it”, and collapsing " +
        "them would hide a decision inside an absence.",
    },
    {
      title: "The edit",
      run: "ctfish-688a7da1ab8ced5a",
      projection: "usage",
      caption:
        "One turn in the longer run writes to the game state rather than playing a move. " +
        "The label attributed to this run comes from the source project's own annotation " +
        "file and is carried verbatim — this tool did not classify the behaviour, it " +
        "imported somebody else's classification and says whose. The trajectory shows " +
        "where in the run it happened; it does not show why, and no forward pass here " +
        "touched the model that did it.",
    },
    {
      title: "And no verification followed it",
      run: "ctfish-688a7da1ab8ced5a",
      projection: "usage",
      caption:
        "The run's verification block reads “ran any verification: no / after the last " +
        "edit: NO”. That is a loop-level fact the adapter can decide from the event " +
        "sequence alone: something was written and nothing afterwards checked it. It is " +
        "the same rule applied to a live Claude Code or Codex session, which is the point " +
        "of putting a foreign corpus through the same vocabulary.",
    },
  ],
});

/* ── 10 · Evaluation awareness ──────────────────────────────────────────────
   The direction, and the null that decides whether it is one. */

export const EVAL_AWARENESS = registerSeerEpisode({
  id: "eval-awareness",
  label: "Does it know it is being tested?",
  blurb:
    "A diff-of-means direction between prompts that look like an evaluation and prompts " +
    "that look like use. Its null cloud decides whether colouring turns by it means " +
    "anything.",
  register: "bench",
  manifest: {
    // The dataset, the direction id and the layer are all the real ones now:
    // the sweep was run on the pinned instruct model and every layer it tried
    // is in that file. L12 is named here because it is the layer whose effect
    // cleared its own null AND transferred to frame pairs the fit never saw —
    // not because it was the largest number in the sweep, which is a different
    // and much weaker reason to pick a layer.
    dataset: "HuggingFaceTB__SmolLM2-135M-Instruct",
    directions: ["eval-awareness-v1-L12"],
    space: "resid.L12",
    unavailable:
      "Needs an `eval-awareness-v1-L12` entry in directions.json — a diff-of-means over " +
      "the frozen eval/deploy matched pairs, written by the §3.2 direction writer " +
      "(`python -m nebulai.backend.eval_awareness`).",
  },
  steps: [
    {
      title: "Two prompt sets that differ in one thing",
      axis: "eval-awareness",
      caption:
        "One set is phrased the way a benchmark item is phrased; the other asks the same " +
        "questions the way a person would. The direction is nothing more than the mean " +
        "residual of the first minus the mean residual of the second, at a pinned layer, " +
        "normalised to length 1. There is no classifier, no judge and no label anybody " +
        "assigned after the fact.",
    },
    {
      title: "What the null actually said",
      axis: "eval-awareness",
      caption:
        "Measured, not assumed. Over 8 framing pairs × 8 tasks on SmolLM2-135M-Instruct " +
        "@12fd25f7, the in-sample effect at layer 12 is d = 1.00 against a " +
        "label-permutation null whose p95 is 0.68, and it survives on two framing pairs " +
        "the fit never saw at d = 0.60. That is the whole of the good news. Of the 30 " +
        "layers swept, 6 cleared their own null (10–13, 15, 16) and 24 did not — layer " +
        "19, the one the persona space uses, came in at d = 0.38 INSIDE its null of 0.66. " +
        "Six of thirty at a per-layer 5% threshold is more than the 1.5 chance would " +
        "give, and it is not a lot more. Every layer tried is in the file.",
    },
    {
      title: "The null cloud, in the same picture",
      axis: "eval-awareness",
      caption:
        "The pale cloud is the same points on a RANDOM unit direction from a fixed seed. " +
        "It is not decoration: without it any direction at all looks like structure, " +
        "because the projection of a high-dimensional cloud onto ANY unit vector has a " +
        "spread. R5 — the null ships with the figure — is why it cannot be turned off.",
    },
    {
      title: "Turns coloured by it",
      axis: "eval-awareness",
      projection: "usage",
      caption:
        "Now the session field, with each turn coloured by its projection onto that " +
        "direction. Read this as a coordinate, not a verdict. A turn high on this axis " +
        "is a turn whose residual has more of this direction in it than another turn " +
        "does; whether that corresponds to anything the model “knows” is a question this " +
        "figure cannot answer and does not claim to.",
    },
    {
      title: "What would make this a causal claim, and why it is not one yet",
      axis: "eval-awareness",
      caption:
        "Adding α times this direction to the residual stream and measuring what changes " +
        "in the output would be an intervention, and an intervention is the one place " +
        "this tool is allowed a causal sentence. That machinery is phase 4. Everything on " +
        "this screen is correlational: a direction, a projection, and a null that says " +
        "how impressed to be.",
    },
  ],
});
