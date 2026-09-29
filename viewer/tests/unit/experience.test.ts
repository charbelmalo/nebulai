import { describe, expect, it } from "vitest";
import {
  defaultPage,
  experienceFromPath,
  inferExperience,
  isExperience,
  resolveExperience,
  supports,
  type RouteIntent,
} from "../../src/app/experience";

describe("experience routing (PRODUCT-EXPERIENCES.md, URL and migration rules)", () => {
  it("recognises only the three experiences", () => {
    expect(isExperience("learn")).toBe(true);
    expect(isExperience("atlas")).toBe(true);
    expect(isExperience("research")).toBe(true);
    expect(isExperience("seer")).toBe(false);
    expect(isExperience("../learn")).toBe(false);
    expect(isExperience(undefined)).toBe(false);
  });

  it("opens each experience on its own page", () => {
    expect(defaultPage("learn")).toBe("guide");
    expect(defaultPage("atlas")).toBe("map");
    expect(defaultPage("research")).toBe("interp");
    // a view or model with no page is a map link, even in Research
    expect(defaultPage("research", { view: "chord" })).toBe("map");
    expect(defaultPage("research", { model: true })).toBe("map");
  });

  it("a guided episode or lesson always selects Learn", () => {
    for (const entry of [null, "atlas", "research", "learn"] as const) {
      const r = resolveExperience({ entry, explicit: "research", intent: { episode: true, page: "interp" } });
      expect(r.experience).toBe("learn");
    }
    expect(resolveExperience({ entry: "atlas", intent: { lesson: true, page: "map" } }).experience).toBe("learn");
  });

  it("an explicit experience wins when it supports the content", () => {
    const r = resolveExperience({ entry: "atlas", explicit: "research", intent: { page: "map", view: "atlas" } });
    expect(r).toEqual({ experience: "research", notice: null });
  });

  it("an explicit experience that cannot host the content is corrected, with a notice", () => {
    const r = resolveExperience({ entry: "learn", explicit: "learn", intent: { page: "map" } });
    expect(r.experience).toBe("atlas");
    expect(r.notice).toBe("Opened in Atlas for this map.");
    const a = resolveExperience({ entry: "atlas", explicit: "atlas", intent: { page: "interp" } });
    expect(a.experience).toBe("research");
    expect(a.notice).toBe("Opened in Research for this analysis.");
  });

  it("unknown experience values are ignored, never used as paths", () => {
    const r = resolveExperience({ entry: "atlas", explicit: "../../etc", intent: {} });
    expect(r).toEqual({ experience: "atlas", notice: null });
  });

  it("the entry path supplies the default only without a conflicting request", () => {
    expect(resolveExperience({ entry: "research", intent: {} }).experience).toBe("research");
    expect(resolveExperience({ entry: "research", intent: { page: "map" } }).experience).toBe("research");
    const r = resolveExperience({ entry: "atlas", intent: { page: "behavior" } });
    expect(r.experience).toBe("research");
    expect(r.notice).not.toBeNull();
  });

  it("root with no recognised intent shows the chooser", () => {
    expect(resolveExperience({ entry: null, intent: {} })).toEqual({ experience: null, notice: null });
  });

  it("root forwards legacy links deterministically, without a notice", () => {
    const cases: [RouteIntent, string][] = [
      [{ page: "interp" }, "research"],
      [{ page: "behavior" }, "research"],
      [{ page: "map", view: "chord" }, "research"],
      [{ view: "hierarchy" }, "research"],
      [{ view: "compare" }, "research"],
      [{ page: "guide" }, "learn"],
      [{ page: "map" }, "atlas"],
      [{ model: true }, "atlas"],
      [{ finding: true }, "atlas"],
      [{ episode: true }, "learn"],
    ];
    for (const [intent, want] of cases) {
      const r = resolveExperience({ entry: null, intent });
      expect(r.experience, JSON.stringify(intent)).toBe(want);
      expect(r.notice).toBeNull();
    }
  });

  it("Learn hosts map and Internals only inside a lesson", () => {
    expect(supports("learn", { page: "map" })).toBe(false);
    expect(supports("learn", { page: "interp" })).toBe(false);
    expect(supports("learn", { model: true })).toBe(false);
    expect(supports("learn", { page: "map", episode: true })).toBe(true);
    expect(supports("learn", { page: "guide" })).toBe(true);
  });

  it("Atlas never hosts advanced views, Internals or Behavior", () => {
    expect(supports("atlas", { page: "map", view: "chord" })).toBe(false);
    expect(supports("atlas", { page: "interp" })).toBe(false);
    expect(supports("atlas", { page: "behavior" })).toBe(false);
    expect(supports("atlas", { page: "map", view: "atlas", model: true, finding: true })).toBe(true);
  });

  it("legacy inference matches the documented order", () => {
    expect(inferExperience({ page: "guide", episode: true })).toBe("learn");
    expect(inferExperience({ page: "guide" })).toBe("learn");
    expect(inferExperience({})).toBeNull();
  });

  it("reads the experience from nested entry paths under the app root", () => {
    const root = "/psychiX/nebulai-maps/";
    expect(experienceFromPath("/psychiX/nebulai-maps/learn/", root)).toBe("learn");
    expect(experienceFromPath("/psychiX/nebulai-maps/atlas/index.html", root)).toBe("atlas");
    expect(experienceFromPath("/psychiX/nebulai-maps/research/", root)).toBe("research");
    expect(experienceFromPath("/psychiX/nebulai-maps/", root)).toBeNull();
    expect(experienceFromPath("/psychiX/nebulai-maps/index.html", root)).toBeNull();
    expect(experienceFromPath("/elsewhere/learn/", root)).toBeNull();
  });
});
