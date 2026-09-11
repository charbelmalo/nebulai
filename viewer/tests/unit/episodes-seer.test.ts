/** The five run-side episodes: the registry, the manifest gate, and the claims
 *  the captions are and are not allowed to make.
 *
 *  The point of testing prose is narrow and worth stating: three of these
 *  episodes quote counts that came out of a real run in a real store, and one
 *  of them is about a model whose weights nobody here has. A caption that
 *  drifts off those facts is a lie that no type checker catches, so the facts
 *  are asserted here next to the gate that decides whether the episode may be
 *  shown at all.
 */

import { describe, expect, it } from "vitest";
import {
  AI_VILLAGE,
  CHESS_HACKING,
  EVAL_AWARENESS,
  PROJECT_VEND,
  SEER_EPISODES,
  SYDNEY,
  findSeerEpisode,
  registerSeerEpisode,
  seerEpisodeAvailability,
  type SeerEpisode,
  type SeerEpisodeContext,
} from "../../src/chrome/episodes.seer";

/** Everything present, nothing loading. Individual tests take this away. */
const FULL: SeerEpisodeContext = {
  runs: [
    "transcript-sample",
    "village-village-synthetic",
    "ctfish-688a7da1ab8ced5a",
    "ctfish-ec5602c824eb21a6",
    "amongus-game-17",
  ],
  corpora: ["transcript", "ai_village", "ctfish", "amongus"],
  spaces: ["smollm2-135m-instruct@12fd25f77366.personas-v1"],
  studies: ["smollm2-135m-instruct@12fd25f77366.no_exclamation"],
  directionsFor: () => ["eval-awareness-v1-L12"],
  networkAllowed: true,
};

const ctx = (over: Partial<SeerEpisodeContext>): SeerEpisodeContext => ({ ...FULL, ...over });

describe("the registry", () => {
  it("holds the five episodes the plan names, in order", () => {
    expect(SEER_EPISODES.map((e) => e.id)).toEqual([
      "sydney",
      "project-vend",
      "ai-village",
      "chess-hacking",
      "eval-awareness",  // the EPISODE id; its direction id carries the layer
    ]);
  });

  it("replaces by id instead of stacking duplicates on a reload", () => {
    const before = SEER_EPISODES.length;
    const again = registerSeerEpisode({ ...SYDNEY, blurb: "re-registered" });
    expect(SEER_EPISODES.length).toBe(before);
    expect(findSeerEpisode("sydney")?.blurb).toBe("re-registered");
    registerSeerEpisode(SYDNEY); // put it back for the rest of the file
    expect(findSeerEpisode("sydney")?.blurb).toBe(SYDNEY.blurb);
    expect(again.id).toBe("sydney");
  });

  it("gives every episode an id, a label, a blurb and at least one step", () => {
    for (const e of SEER_EPISODES) {
      expect(e.id, e.id).toMatch(/^[a-z0-9-]+$/);
      expect(e.label.length, e.id).toBeGreaterThan(0);
      expect(e.blurb.length, e.id).toBeGreaterThan(0);
      expect(e.steps.length, e.id).toBeGreaterThan(0);
      for (const s of e.steps) {
        expect(s.title.length, `${e.id}/${s.title}`).toBeGreaterThan(0);
        expect(s.caption.length, `${e.id}/${s.title}`).toBeGreaterThan(0);
      }
    }
  });

  it("never points a step at a run the episode's own manifest does not require", () => {
    // a step that fronts a run the gate never checked is a step that can open
    // on an empty plot
    for (const e of SEER_EPISODES) {
      const required = new Set(e.manifest.runs ?? []);
      for (const s of e.steps) {
        if (!s.run) continue;
        expect(required.has(s.run), `${e.id} step "${s.title}" fronts unchecked run ${s.run}`).toBe(
          true,
        );
      }
    }
  });
});

describe("the manifest gate", () => {
  it("lets an episode run when everything it needs is there", () => {
    for (const e of [SYDNEY, PROJECT_VEND, AI_VILLAGE, CHESS_HACKING, EVAL_AWARENESS]) {
      expect(seerEpisodeAvailability(e, FULL).state, e.id).toBe("ready");
    }
  });

  it("says PENDING while the store is still being read, not unavailable", () => {
    const a = seerEpisodeAvailability(SYDNEY, ctx({ runs: null, corpora: null }));
    expect(a.state).toBe("pending");
    if (a.state !== "ready") expect(a.reason).toMatch(/Seer store/);
  });

  it("names the missing run rather than failing vaguely", () => {
    const a = seerEpisodeAvailability(CHESS_HACKING, ctx({ runs: ["ctfish-ec5602c824eb21a6"] }));
    expect(a.state).toBe("unavailable");
    if (a.state !== "ready") {
      expect(a.reason).toContain("ctfish-688a7da1ab8ced5a");
      expect(a.reason).not.toContain("ctfish-ec5602c824eb21a6");
      expect(a.reason).toContain("seer import ctfish"); // and says how to fix it
    }
  });

  it("blames the network before it blames the missing run", () => {
    // a static deploy has neither; only one of those is the real reason
    const a = seerEpisodeAvailability(AI_VILLAGE, ctx({ networkAllowed: false, corpora: [] }));
    expect(a.state).toBe("unavailable");
    if (a.state !== "ready") {
      expect(a.reason).toMatch(/live fetch/);
      expect(a.reason).toMatch(/research-only/);
      expect(a.reason).toMatch(/never vendored|never written into this repo/);
    }
  });

  it("refuses the eval-awareness episode when the direction is not in the sidecar", () => {
    const a = seerEpisodeAvailability(EVAL_AWARENESS, ctx({ directionsFor: () => ["sentiment"] }));
    expect(a.state).toBe("unavailable");
    if (a.state !== "ready") expect(a.reason).toContain("eval-awareness-v1-L12");
  });

  it("waits rather than refusing when the directions sidecar has not loaded", () => {
    const a = seerEpisodeAvailability(EVAL_AWARENESS, ctx({ directionsFor: () => null }));
    expect(a.state).toBe("pending");
    if (a.state !== "ready")
      expect(a.reason).toContain("HuggingFaceTB__SmolLM2-135M-Instruct/directions.json");
  });

  it("does not consult the network flag for an episode that never fetches", () => {
    expect(seerEpisodeAvailability(SYDNEY, ctx({ networkAllowed: false })).state).toBe("ready");
    expect(seerEpisodeAvailability(CHESS_HACKING, ctx({ networkAllowed: false })).state).toBe(
      "ready",
    );
  });

  it("passes an empty manifest, because an episode that needs nothing needs nothing", () => {
    const bare: SeerEpisode = {
      id: "bare",
      label: "l",
      blurb: "b",
      register: "story",
      manifest: {},
      steps: [{ title: "t", caption: "c" }],
    };
    expect(
      seerEpisodeAvailability(bare, ctx({ runs: null, corpora: null, spaces: null, studies: null }))
        .state,
    ).toBe("ready");
  });
});

describe("what the Sydney captions are required to say", () => {
  const all = SYDNEY.steps.map((s) => s.caption).join("\n");

  it("says in words that the placement is NOT model-internal", () => {
    expect(all).toContain("NOT model-internal");
    expect(all).toMatch(/text embedder/);
  });

  it("states that the weights are closed and will not be obtained", () => {
    expect(all).toMatch(/closed/);
  });

  it("never renders the absent usage as zero", () => {
    expect(all).toMatch(/missing is not zero|agent reported no usage/);
    expect(all).not.toMatch(/\b0 tokens\b/);
  });

  it("declares the reconstructed span as reconstructed", () => {
    expect(all).toMatch(/reconstructed/);
  });

  it("declines the persona conclusion the picture invites", () => {
    expect(all).toMatch(/Not that the model was in a persona/);
  });

  it("files itself under the text-embedder space, not a model space", () => {
    expect(SYDNEY.manifest.space).toBe("text_embedder");
  });
});

describe("the numbers the captions quote", () => {
  // these came out of `seer show` against the store the fixtures import into.
  // If an adapter changes what it emits, this test fails before a stale number
  // reaches a reader.
  const caption = (ep: SeerEpisode, i: number) => ep.steps[i]!.caption;

  it("quotes the transcript run's real event count and span", () => {
    expect(caption(SYDNEY, 0)).toContain("16 events");
    expect(caption(SYDNEY, 0)).toContain("12.0 s");
    expect(caption(SYDNEY, 0)).toMatch(/RECONCILED/);
  });

  it("quotes both ctfish runs' real counts, including the single edit", () => {
    const c = caption(CHESS_HACKING, 0);
    expect(c).toContain("65.0 s");
    expect(c).toContain("69 events");
    expect(c).toContain("114.0 s");
    expect(c).toContain("118 events");
    expect(c).toMatch(/edit ×1/);
  });

  it("quotes the dropped-by-policy count verbatim and keeps it distinct from missing", () => {
    const c = caption(CHESS_HACKING, 1);
    expect(c).toContain("21");
    expect(c).toContain("ctfish.THOUGHT");
    expect(c).toMatch(/dropped_by_policy/);
    expect(c).toMatch(/never folds it into `missing`|never folds it into missing/);
  });

  it("quotes the verification line the adapter actually printed", () => {
    expect(caption(CHESS_HACKING, 3)).toMatch(/after the last edit: NO/);
  });

  it("labels the AI Village fixture as synthetic and gives its real size", () => {
    const c = caption(AI_VILLAGE, 1);
    expect(c).toContain("14-event");
    expect(c).toContain("240 s");
    expect(c).toMatch(/SYNTHETIC/);
    expect(c).toMatch(/not AI Village data/);
  });
});

describe("what the analytical episodes refuse to claim", () => {
  it("makes the ensemble episode about the band, not the median", () => {
    const all = PROJECT_VEND.steps.map((s) => s.caption).join("\n");
    expect(all).toMatch(/Read the width, not the line/);
    expect(all).toMatch(/N = 20/); // the interval-not-a-point rule
    expect(all).toMatch(/anecdote/);
  });

  it("keeps the eval-awareness episode correlational and ships its null", () => {
    const all = EVAL_AWARENESS.steps.map((s) => s.caption).join("\n");
    expect(all).toMatch(/null/);
    expect(all).toMatch(/R5/);
    expect(all).toMatch(/correlational/);
    expect(all).toMatch(/coordinate, not a verdict/);
    // and it does not smuggle in a judge
    expect(all).toMatch(/no classifier, no judge/i);
  });

  it("records the space every eval-awareness quantity lives in", () => {
    expect(EVAL_AWARENESS.manifest.space).toBe("resid.L12");
    // the layer in the space, the direction id and the caption must agree: a
    // figure that names one layer and reads another is the quietest possible lie
    expect(EVAL_AWARENESS.manifest.directions).toEqual(["eval-awareness-v1-L12"]);
    const captions = EVAL_AWARENESS.steps.map((s) => s.caption).join("\n");
    expect(captions).toContain("layer 12");
    // the sweep is stated, not hidden: 6 of 30 cleared, 24 did not
    expect(captions).toContain("30 ");
    expect(captions).toContain("24 did not");
  });
});
