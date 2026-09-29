/** Episodes (P5) — the registry, the availability gate, and the one rule the
 *  gate exists to enforce: an episode narrates exact numbers out of one named
 *  artifact, so when that artifact is absent it must say what is missing rather
 *  than run with the numbers gone or quietly swap in another model (§2.2).
 *
 *  `applyTourStep` is exercised against the real store, because the whole
 *  premise of a tour step is that it goes through the SAME actions a user's
 *  clicks would fire — a step that wrote state by another route would be a
 *  state no permalink could reproduce.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { appStore } from "../../src/app/store";
import {
  TOURS,
  applyTourStep,
  episodeAvailability,
  findTour,
  register,
  type EpisodeContext,
  type Tour,
} from "../../src/chrome/tours";
import { GROKKING, SOLID_GOLD_MAGIKARP } from "../../src/chrome/episodes";

/** What a fully-stocked deploy can see. Individual tests take this away one
 *  piece at a time — that removal is the actual subject of most of them. */
function ctx(over: Partial<EpisodeContext> = {}): EpisodeContext {
  return {
    datasets: ["gpt2", "gpt2-medium"],
    channelsFor: () => ["we_norm", "we_centroid_dist"],
    channelsLoaded: () => true,
    features: [
      "comp-web",
      "head-fingerprints",
      "induction-microscope",
      "ablation-ghosts",
      "logit-attrib",
      "causal-patching",
      "occlusion-vignette",
      "sae-decoder",
      "sae-piano-roll",
      "decoder-cosine-web",
      "direction-compass",
      "grokking-clock",
      "fourier-atlas",
    ],
    ...over,
  };
}

describe("the registry", () => {
  it("registers the new episodes exactly once, by id", () => {
    expect(findTour("glitch-knot")).toBe(SOLID_GOLD_MAGIKARP);
    expect(findTour("grokking-clock")).toBe(GROKKING);
    expect(TOURS.filter((t) => t.id === "glitch-knot")).toHaveLength(1);
  });

  it("replaces rather than appends when an id is re-registered", () => {
    const n = TOURS.length;
    const replacement: Tour = {
      id: "glitch-knot",
      label: "x",
      blurb: "x",
      model: "gpt2",
      steps: [{ page: "map", title: "t", caption: "c" }],
    };
    register(replacement);
    expect(TOURS).toHaveLength(n);
    expect(findTour("glitch-knot")).toBe(replacement);
    // put the real one back — TOURS is module state shared with the next test
    register(SOLID_GOLD_MAGIKARP);
    expect(findTour("glitch-knot")).toBe(SOLID_GOLD_MAGIKARP);
  });

  it("gives the three pre-registry tours manifests too", () => {
    for (const id of ["induction", "ioi", "sae-feature"]) {
      const t = findTour(id)!;
      expect(t.manifest, id).toBeDefined();
      expect(t.manifest!.features!.length, id).toBeGreaterThan(0);
    }
  });

  it("keeps every episode's steps non-empty and captioned", () => {
    for (const t of TOURS) {
      expect(t.steps.length, t.id).toBeGreaterThan(0);
      for (const s of t.steps) {
        expect(s.title, t.id).toBeTruthy();
        expect(s.caption.length, t.id).toBeGreaterThan(20);
      }
    }
  });
});

describe("episodeAvailability", () => {
  it("plays an episode whose artifacts are all present", () => {
    expect(episodeAvailability(SOLID_GOLD_MAGIKARP, ctx())).toEqual({ state: "ready" });
    expect(episodeAvailability(GROKKING, ctx())).toEqual({ state: "ready" });
  });

  it("treats a tour with no manifest as ready — it makes no artifact claim", () => {
    const bare: Tour = { id: "x", label: "x", blurb: "x", model: "gpt2", steps: [] };
    expect(episodeAvailability(bare, ctx())).toEqual({ state: "ready" });
  });

  it("names the missing map, and does not offer another one", () => {
    const av = episodeAvailability(SOLID_GOLD_MAGIKARP, ctx({ datasets: ["gpt2-medium"] }));
    expect(av.state).toBe("unavailable");
    expect(av).toHaveProperty("reason");
    const reason = (av as { reason: string }).reason;
    expect(reason).toContain("gpt2");
    // the pinned model is never substituted: the other map on the shelf is not
    // proposed as a stand-in anywhere in the sentence
    expect(reason).not.toContain("instead");
  });

  it("names the missing channels AND the command that writes them", () => {
    const av = episodeAvailability(
      SOLID_GOLD_MAGIKARP,
      ctx({ channelsFor: () => ["we_norm"] }),
    );
    expect(av.state).toBe("unavailable");
    const reason = (av as { reason: string }).reason;
    expect(reason).toContain("we_centroid_dist");
    expect(reason).not.toContain("we_norm,");
    expect(reason).toContain("nebulai channels gpt2");
  });

  it("separates a sidecar in flight from a sidecar that is not there", () => {
    const inFlight = episodeAvailability(
      SOLID_GOLD_MAGIKARP,
      ctx({ channelsFor: () => null, channelsLoaded: () => false }),
    );
    expect(inFlight.state).toBe("pending");

    const absent = episodeAvailability(
      SOLID_GOLD_MAGIKARP,
      ctx({ channelsFor: () => null, channelsLoaded: () => true }),
    );
    expect(absent.state).toBe("unavailable");
  });

  it("names the missing Internals view", () => {
    const av = episodeAvailability(GROKKING, ctx({ features: ["fourier-atlas"] }));
    expect(av.state).toBe("unavailable");
    expect((av as { reason: string }).reason).toContain("grokking-clock");
  });

  it("carries the episode's own explanation into every refusal", () => {
    const av = episodeAvailability(SOLID_GOLD_MAGIKARP, ctx({ datasets: [] }));
    expect((av as { reason: string }).reason).toContain(
      SOLID_GOLD_MAGIKARP.manifest!.unavailable!,
    );
  });
});

describe("the SolidGoldMagikarp episode", () => {
  it("declares the raw embedding space, which is NOT the map's own layout", () => {
    // the map is laid out on W_E.centered; every number the captions quote is
    // a length in W_E.raw. Tagging them the same would invite exactly the
    // comparison D2 refuses.
    expect(SOLID_GOLD_MAGIKARP.manifest!.space).toBe("W_E.raw");
  });

  it("tells every step on the map page, against the pinned model", () => {
    for (const s of SOLID_GOLD_MAGIKARP.steps) {
      expect(s.page).toBe("map");
      expect(s.dataset).toBe("gpt2");
      expect(s.feature).toBeUndefined();
    }
  });

  it("quotes the measured numbers, not the folklore", () => {
    const text = SOLID_GOLD_MAGIKARP.steps.map((s) => s.caption).join(" ");
    // the knot's extreme value and the 1st percentile it sits below
    expect(text).toContain("1.534");
    expect(text).toContain("2.551");
    // and the correction: the famous token is nowhere near the knot
    expect(text).toContain("3.177");
    expect(text).toContain("SolidGoldMagikarp");
  });

  it("filters on the raw units the caption quotes", () => {
    const windowed = SOLID_GOLD_MAGIKARP.steps.filter((s) => s.channelWindow);
    expect(windowed.length).toBeGreaterThan(0);
    for (const s of windowed) {
      const [lo, hi] = s.channelWindow!;
      expect(lo).toBeLessThan(hi);
      // the channel's real range is 1.534–5.570; a window in 0–1 would mean
      // the episode was narrating a normalised restatement
      expect(hi).toBeGreaterThan(1.5);
    }
  });
});

describe("the grokking episode", () => {
  it("carries no space tag at all", () => {
    // R7: the toy model's hidden layer is not one of GPT-2's, and tagging it
    // `resid.L…` would place it in a basis it has no business being compared in
    expect(GROKKING.manifest!.space).toBeUndefined();
  });

  it("says in its first caption that this is not GPT-2", () => {
    expect(GROKKING.steps[0]!.caption).toContain("not");
    expect(GROKKING.blurb).toContain("not GPT-2");
  });
});

describe("applyTourStep", () => {
  beforeEach(() => {
    const st = appStore.getState();
    st.setChannel(null);
    st.setMapQuery("");
    st.setSelection(null);
  });

  it("lights the lens with the window the caption quotes", () => {
    const at = SOLID_GOLD_MAGIKARP.steps.findIndex((s) => s.channelWindow);
    applyTourStep(SOLID_GOLD_MAGIKARP, at);
    const s = appStore.getState();
    expect(s.channel.id).toBe(SOLID_GOLD_MAGIKARP.steps[at]!.channel);
    expect(s.channel.window).toEqual(SOLID_GOLD_MAGIKARP.steps[at]!.channelWindow);
  });

  it("puts the lens away when a step asks it to", () => {
    applyTourStep(SOLID_GOLD_MAGIKARP, 1);
    expect(appStore.getState().channel.id).toBe("we_norm");
    applyTourStep(SOLID_GOLD_MAGIKARP, 0);
    expect(appStore.getState().channel.id).toBeNull();
  });

  it("leaves the lens alone for a step that does not mention it", () => {
    const st = appStore.getState();
    st.setChannel("we_norm", [2, 3]);
    // an interp-page step: nothing in it addresses the map's lens
    applyTourStep(findTour("induction")!, 0);
    expect(appStore.getState().channel.id).toBe("we_norm");
  });

  it("pins the point the caption names, and runs its search", () => {
    const at = SOLID_GOLD_MAGIKARP.steps.findIndex((s) => s.search);
    applyTourStep(SOLID_GOLD_MAGIKARP, at);
    const s = appStore.getState();
    expect(s.mapQuery.text).toBe(SOLID_GOLD_MAGIKARP.steps[at]!.search);
    expect(s.selection).toEqual(SOLID_GOLD_MAGIKARP.steps[at]!.mapSelection);
  });

  it("is a no-op for a step index that does not exist", () => {
    expect(() => applyTourStep(SOLID_GOLD_MAGIKARP, 99)).not.toThrow();
    expect(appStore.getState().channel.id).toBeNull();
  });
});
