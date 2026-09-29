/** Learn's introductory lesson: completion is judged only from the reader's
 *  own choices, URL keys are validated, availability needs the manifest's
 *  exact artifact, and progress never crosses artifact digests. */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ExperienceManifest } from "../../src/data/experience";
import {
  completedLessons,
  emptyProgress,
  findStepOf,
  incompleteSteps,
  lessonAvailability,
  lessonIdOf,
  lessonTourId,
  loadProgress,
  markCompleted,
  needsUnit,
  parseLessonKeys,
  saveProgress,
  stepDone,
  WHAT_IS_A_POINT as L,
} from "../../src/chrome/learn/lesson";
import { findTour } from "../../src/chrome/tours";

const SHA = "d".repeat(64);
const labels: Record<number, string> = { 0: "numbers and quantities", 1: "words related to piers" };
const labelOf = (row: number) => labels[row] ?? null;

function fakeStorage(): Storage {
  const m = new Map<string, string>();
  return {
    get length() {
      return m.size;
    },
    clear: () => m.clear(),
    getItem: (k) => m.get(k) ?? null,
    key: (i) => [...m.keys()][i] ?? null,
    removeItem: (k) => void m.delete(k),
    setItem: (k, v) => void m.set(k, String(v)),
  };
}

describe("the lesson tour", () => {
  it("is registered under a namespaced id with five steps on the map page", () => {
    expect(L.id).toBe("lesson:what-is-a-point");
    expect(findTour(L.id)).toBe(L);
    expect(L.kind).toBe("lesson");
    expect(L.steps).toHaveLength(5);
    expect(L.steps.every((s) => s.page === "map")).toBe(true);
    expect(lessonIdOf(L.id)).toBe("what-is-a-point");
    expect(lessonIdOf("induction")).toBeNull();
    expect(lessonTourId("x")).toBe("lesson:x");
  });

  it("has exactly one correct answer per check, each with an explanation", () => {
    for (const s of L.steps) {
      if (s.task?.kind !== "check") continue;
      expect(s.task.options.filter((o) => o.correct)).toHaveLength(1);
      for (const o of s.task.options) expect(o.why.length).toBeGreaterThan(40);
    }
  });
});

describe("completion (pure)", () => {
  const find = findStepOf(L);

  it("the orientation step has no task and is never listed as incomplete", () => {
    expect(stepDone(L, 0, emptyProgress(SHA), labelOf)).toBe(true);
    expect(incompleteSteps(L, emptyProgress(SHA), labelOf)).toEqual([1, 2, 3, 4]);
  });

  it("the find step needs a unit whose label matches, not any unit", () => {
    expect(find).toBe(1);
    expect(stepDone(L, find, { ...emptyProgress(SHA), unitRow: 1 }, labelOf)).toBe(false);
    expect(stepDone(L, find, { ...emptyProgress(SHA), unitRow: 0 }, labelOf)).toBe(true);
    // an unlabelled map's placeholder never satisfies it
    expect(stepDone(L, find, { ...emptyProgress(SHA), unitRow: 0 }, () => null)).toBe(false);
  });

  it("a check step is done only with its correct answer", () => {
    const p = emptyProgress(SHA);
    expect(stepDone(L, 2, { ...p, answers: { 2: "label" } }, labelOf)).toBe(false);
    expect(stepDone(L, 2, { ...p, answers: { 2: "index" } }, labelOf)).toBe(true);
    expect(stepDone(L, 3, { ...p, answers: { 3: "activation" } }, labelOf)).toBe(false);
    expect(stepDone(L, 3, { ...p, answers: { 3: "cause" } }, labelOf)).toBe(false);
    expect(stepDone(L, 3, { ...p, answers: { 3: "geometry" } }, labelOf)).toBe(true);
    // an answer recorded against another step does not count
    expect(stepDone(L, 3, { ...p, answers: { 2: "geometry" } }, labelOf)).toBe(false);
  });

  it("the finish step is done only when finished explicitly", () => {
    const all = { ...emptyProgress(SHA), unitRow: 0, answers: { 2: "index", 3: "geometry" } };
    expect(incompleteSteps(L, all, labelOf)).toEqual([4]);
    expect(incompleteSteps(L, { ...all, finished: [4] }, labelOf)).toEqual([]);
  });

  it("every step after the find step is about the found unit", () => {
    expect([0, 1, 2, 3, 4].map((i) => needsUnit(L, i))).toEqual([false, false, true, true, true]);
  });
});

describe("URL keys", () => {
  it("accept a registered lesson and clamp the step", () => {
    expect(parseLessonKeys("what-is-a-point", "2", findTour)).toEqual({ tourId: L.id, step: 2 });
    expect(parseLessonKeys("what-is-a-point", null, findTour)).toEqual({ tourId: L.id, step: 0 });
    expect(parseLessonKeys("what-is-a-point", "99", findTour)).toEqual({ tourId: L.id, step: 4 });
    expect(parseLessonKeys("what-is-a-point", "-1", findTour)).toEqual({ tourId: L.id, step: 0 });
    expect(parseLessonKeys("what-is-a-point", "1.5", findTour)).toEqual({ tourId: L.id, step: 0 });
  });

  it("reject unknown, malformed and non-lesson ids", () => {
    expect(parseLessonKeys(null, "0", findTour)).toBeNull();
    expect(parseLessonKeys("nope", "0", findTour)).toBeNull();
    expect(parseLessonKeys("../x", "0", findTour)).toBeNull();
    expect(parseLessonKeys("What-Is-A-Point", "0", findTour)).toBeNull();
    // an episode id is not a lesson, even though it is a registered tour
    expect(parseLessonKeys("induction", "0", findTour)).toBeNull();
  });
});

describe("availability", () => {
  const manifest = (over: Partial<ExperienceManifest> = {}): ExperienceManifest =>
    ({
      schema_version: 1,
      learn_intro: { lesson_id: "what-is-a-point", dataset_id: L.manifest!.dataset!, sha256: SHA },
      artifacts: [{ dataset_id: L.manifest!.dataset!, sha256: SHA, path: `artifacts/${SHA}/nebulai.json` }],
      ...over,
    }) as unknown as ExperienceManifest;
  const ds = [L.manifest!.dataset!];

  it("is ready only on the manifest's exact artifact", () => {
    expect(lessonAvailability(L, { state: "ok", manifest: manifest() }, ds)).toEqual({ state: "ready", sha256: SHA });
  });

  it("waits while the manifest loads, and refuses without one", () => {
    expect(lessonAvailability(L, { state: "unloaded" }, ds).state).toBe("pending");
    expect(lessonAvailability(L, { state: "absent" }, ds).state).toBe("unavailable");
    expect(lessonAvailability(L, { state: "invalid", errors: ["x"] }, ds).state).toBe("unavailable");
  });

  it("refuses a manifest that names another lesson, map or unlisted digest", () => {
    const li = manifest().learn_intro;
    const other = (x: object) => ({ state: "ok" as const, manifest: manifest({ learn_intro: { ...li, ...x } }) });
    expect(lessonAvailability(L, other({ lesson_id: "other" }), ds).state).toBe("unavailable");
    expect(lessonAvailability(L, other({ dataset_id: "gpt2" }), ds).state).toBe("unavailable");
    expect(lessonAvailability(L, other({ sha256: "e".repeat(64) }), ds).state).toBe("unavailable");
  });

  it("refuses when the index does not carry the map", () => {
    expect(lessonAvailability(L, { state: "ok", manifest: manifest() }, ["gpt2"]).state).toBe("unavailable");
  });
});

describe("storage", () => {
  beforeEach(() => {
    Object.assign(globalThis, { sessionStorage: fakeStorage(), localStorage: fakeStorage() });
  });
  afterEach(() => {
    delete (globalThis as { sessionStorage?: Storage }).sessionStorage;
    delete (globalThis as { localStorage?: Storage }).localStorage;
  });

  it("round-trips progress for the same digest and drops it for another", () => {
    const p = { sha256: SHA, unitRow: 0, answers: { 2: "index" }, finished: [] };
    saveProgress("what-is-a-point", p);
    expect(loadProgress("what-is-a-point", SHA)).toEqual(p);
    expect(loadProgress("what-is-a-point", "e".repeat(64))).toEqual(emptyProgress("e".repeat(64)));
  });

  it("sanitises a tampered record", () => {
    sessionStorage.setItem(
      "nebulai.lesson.what-is-a-point",
      JSON.stringify({ sha256: SHA, unitRow: -3, answers: "x", finished: [4, "a"] }),
    );
    expect(loadProgress("what-is-a-point", SHA)).toEqual({ sha256: SHA, unitRow: null, answers: {}, finished: [4] });
    sessionStorage.setItem("nebulai.lesson.what-is-a-point", "{not json");
    expect(loadProgress("what-is-a-point", SHA)).toEqual(emptyProgress(SHA));
  });

  it("records completion once", () => {
    markCompleted("what-is-a-point");
    markCompleted("what-is-a-point");
    expect(completedLessons()).toEqual(["what-is-a-point"]);
  });

  it("keeps working when storage throws", () => {
    const boom = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
    Object.assign(globalThis, { sessionStorage: boom, localStorage: boom });
    expect(() => saveProgress("what-is-a-point", emptyProgress(SHA))).not.toThrow();
    expect(loadProgress("what-is-a-point", SHA)).toEqual(emptyProgress(SHA));
    expect(() => markCompleted("what-is-a-point")).not.toThrow();
    expect(completedLessons()).toEqual([]);
  });
});
