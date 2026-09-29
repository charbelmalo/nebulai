/** learn/lesson.ts — NebulAI Learn's introductory lesson, "What is a point?",
 *  and the rules that decide when a lesson step is done.
 *
 *  The lesson is a registered tour (chrome/tours.ts), not a second runner: it
 *  is stepped by the same `runEpisodeStep`, restored by the same permalink
 *  path and shown progress by the same header. What it adds is reader tasks.
 *  A step with a task is complete only when the reader's own selection or
 *  answer satisfies it — never because the step was shown, a timer ran or an
 *  animation ended.
 *
 *  Identity: the tour id is namespaced (`lesson:what-is-a-point`) so it can
 *  never collide with an episode id, and the URL keys `lesson` /
 *  `lesson_step` are validated against this registry before anything runs.
 *
 *  Progress lives in sessionStorage keyed by lesson and artifact digest, so a
 *  reload restores it and a different map never inherits it. Completion lives
 *  in localStorage (a per-viewer convenience; nothing is shared). Both are
 *  wrapped because storage can be blocked, and the lesson must still work. */

import { findArtifact, type ManifestStatus } from "../../data/experience";
import { register, type LessonOption, type LessonTask, type Tour } from "../tours";

export const LESSON_PREFIX = "lesson:";
export const INTRO_LESSON = "what-is-a-point";
/** the map the introductory lesson's words describe */
export const INTRO_DATASET = "gpt2-small__sae__blocks.8.hook_resid_pre";

const LESSON_ID_RE = /^[a-z0-9-]{1,64}$/;

export function lessonTourId(lessonId: string): string {
  return `${LESSON_PREFIX}${lessonId}`;
}

/** `lesson:what-is-a-point` → `what-is-a-point`; null for an episode id */
export function lessonIdOf(tourId: string): string | null {
  return tourId.startsWith(LESSON_PREFIX) ? tourId.slice(LESSON_PREFIX.length) : null;
}

export function isLessonTour(t: Tour | undefined | null): boolean {
  return !!t && t.kind === "lesson";
}

/* ── the lesson ───────────────────────────────────────────────────────── */

const IDENTITY_OPTIONS: LessonOption[] = [
  {
    id: "label",
    text: "Its label",
    correct: false,
    why:
      "A label is a description written by another tool (here, Neuronpedia). Several units can carry " +
      "similar labels, and a label can be rewritten without the unit changing.",
  },
  {
    id: "index",
    text: "Its index in this SAE, together with the map file's digest",
    correct: true,
    why:
      "The unit kind names the SAE and the layer it reads, the index names one direction inside it, and " +
      "the sha256 digest names the exact file. Together they point to one unit that anyone can find again.",
  },
  {
    id: "position",
    text: "Where it sits on the screen",
    correct: false,
    why:
      "Positions are projection coordinates. Building the layout again can move every point while every " +
      "unit stays the same.",
  },
];

const MEANING_OPTIONS: LessonOption[] = [
  {
    id: "activation",
    text: "This direction was strongly active on text I entered",
    correct: false,
    why:
      "Nothing you type reaches this map. It was built from the SAE's decoder vectors, so it holds no " +
      "activation on any input, yours or anyone else's.",
  },
  {
    id: "geometry",
    text: "Its decoder vector points a similar way to its neighbours' decoder vectors",
    correct: true,
    why:
      "That is what the layout encodes: how similar the decoder vectors are, flattened into two " +
      "dimensions. It is a relationship between learned weights, not a measurement on text.",
  },
  {
    id: "cause",
    text: "This direction makes GPT-2 write about numbers",
    correct: false,
    why:
      "The map shows no intervention. A causal claim needs an experiment that changes the model's " +
      "forward pass and measures what happens; this map records neither.",
  },
];

export const WHAT_IS_A_POINT: Tour = register({
  id: lessonTourId(INTRO_LESSON),
  kind: "lesson",
  label: "What is a point?",
  blurb:
    "Find one direction on a real map, read what identifies it exactly, and tell its label and " +
    "geometry apart from what it does not show. About five minutes.",
  model: INTRO_DATASET,
  manifest: {
    dataset: INTRO_DATASET,
    unavailable: "This lesson reads the published starter map and checks it against the release manifest.",
  },
  steps: [
    {
      page: "map",
      dataset: INTRO_DATASET,
      search: "",
      mapSelection: null,
      title: "A map of learned directions",
      caption:
        "Each point is one direction that a sparse autoencoder (SAE) learned to read from GPT-2 Small's " +
        "residual stream at layer 8. Points sit close together when their decoder vectors are similar. " +
        "The panel names the exact file you are looking at.",
    },
    {
      page: "map",
      title: "Find a direction about numbers",
      caption:
        "Search the labels, then choose one unit whose label is about numbers. You can also pick it on the " +
        "map. The labels come from Neuronpedia, which describes the text each direction responds to.",
      task: {
        kind: "find",
        match: "number|numer|digit|quantit",
        hint: "Type “number” in the search box, then choose a unit whose label mentions numbers.",
      },
    },
    {
      page: "map",
      title: "What identifies this unit?",
      caption:
        "The unit panel lists everything recorded about the unit you chose. Read it, then " +
        "answer: if you sent this unit to someone else, what would let them find exactly the same one?",
      task: {
        kind: "check",
        question: "Which of these identifies this unit exactly?",
        options: IDENTITY_OPTIONS,
      },
    },
    {
      page: "map",
      title: "What does its place on the map mean?",
      caption:
        "Your unit sits near other directions. The map was built only from the SAE's weights, and its " +
        "labels were written separately.",
      task: {
        kind: "check",
        question: "This unit sits near other number-related directions. What does that tell you?",
        options: MEANING_OPTIONS,
      },
    },
    {
      page: "map",
      title: "Keep it, or keep exploring",
      caption:
        "Save this unit as a finding file, or open it in Atlas and explore the whole map on your own. " +
        "Either way it is identified exactly as in step 3. Your answers here are not saved or shared.",
      task: { kind: "finish" },
    },
  ],
});

/* ── completion rules (pure) ──────────────────────────────────────────── */

export interface LessonProgress {
  /** the artifact this progress was made on; progress never crosses digests */
  sha256: string;
  /** the unit the reader found in the find step */
  unitRow: number | null;
  /** chosen option per check step */
  answers: Record<number, string>;
  /** finish steps the reader completed explicitly */
  finished: number[];
}

export function emptyProgress(sha256: string): LessonProgress {
  return { sha256, unitRow: null, answers: {}, finished: [] };
}

/** Does a label satisfy a find task? Case-insensitive, whole label. */
export function labelMatches(task: Extract<LessonTask, { kind: "find" }>, label: string | null): boolean {
  if (!label) return false;
  return new RegExp(task.match, "i").test(label);
}

/** Is this one step complete, judged ONLY from what the reader did.
 *  `labelOf` reads a row's label from the loaded (verified) dataset. */
export function stepDone(
  tour: Tour,
  step: number,
  p: LessonProgress,
  labelOf: (row: number) => string | null,
): boolean {
  const task = tour.steps[step]?.task;
  if (!task) return true;
  switch (task.kind) {
    case "find":
      return p.unitRow !== null && labelMatches(task, labelOf(p.unitRow));
    case "check": {
      const chosen = task.options.find((o) => o.id === p.answers[step]);
      return !!chosen?.correct;
    }
    case "finish":
      return p.finished.includes(step);
  }
}

/** Steps whose task is not yet done, in order (steps without a task are
 *  never listed). */
export function incompleteSteps(tour: Tour, p: LessonProgress, labelOf: (row: number) => string | null): number[] {
  const out: number[] = [];
  tour.steps.forEach((s, i) => {
    if (s.task && !stepDone(tour, i, p, labelOf)) out.push(i);
  });
  return out;
}

/** Does this step need the found unit before it can be shown? Every step
 *  after the find step is about that unit. */
export function needsUnit(tour: Tour, step: number): boolean {
  const find = tour.steps.findIndex((s) => s.task?.kind === "find");
  return find >= 0 && step > find;
}

export function findStepOf(tour: Tour): number {
  return tour.steps.findIndex((s) => s.task?.kind === "find");
}

/** Validate URL lesson keys. Unknown ids are rejected; a missing or invalid
 *  step is 0; a step past the end is clamped to the last step. */
export function parseLessonKeys(
  lesson: string | null,
  step: string | null,
  known: (tourId: string) => Tour | undefined,
): { tourId: string; step: number } | null {
  if (!lesson || !LESSON_ID_RE.test(lesson)) return null;
  const tour = known(lessonTourId(lesson));
  if (!tour || tour.kind !== "lesson") return null;
  const n = Number(step ?? "0");
  const s = Number.isInteger(n) && n >= 0 ? Math.min(n, tour.steps.length - 1) : 0;
  return { tourId: tour.id, step: s };
}

/* ── storage ──────────────────────────────────────────────────────────── */

const PROGRESS_KEY = (id: string) => `nebulai.lesson.${id}`;
const DONE_KEY = "nebulai.lessons.completed";

export function loadProgress(lessonId: string, sha256: string): LessonProgress {
  try {
    const raw = sessionStorage.getItem(PROGRESS_KEY(lessonId));
    if (!raw) return emptyProgress(sha256);
    const p = JSON.parse(raw) as Partial<LessonProgress>;
    if (p.sha256 !== sha256) return emptyProgress(sha256);
    return {
      sha256,
      unitRow: typeof p.unitRow === "number" && Number.isInteger(p.unitRow) && p.unitRow >= 0 ? p.unitRow : null,
      answers: p.answers && typeof p.answers === "object" ? (p.answers as Record<number, string>) : {},
      finished: Array.isArray(p.finished) ? p.finished.filter((n): n is number => Number.isInteger(n)) : [],
    };
  } catch {
    return emptyProgress(sha256);
  }
}

export function saveProgress(lessonId: string, p: LessonProgress): void {
  try {
    sessionStorage.setItem(PROGRESS_KEY(lessonId), JSON.stringify(p));
  } catch {
    /* storage blocked: progress lasts for this page only */
  }
}

export function completedLessons(): string[] {
  try {
    const raw = localStorage.getItem(DONE_KEY);
    const list = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(list) ? list.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export function markCompleted(lessonId: string): void {
  try {
    const list = completedLessons();
    if (!list.includes(lessonId)) localStorage.setItem(DONE_KEY, JSON.stringify([...list, lessonId]));
  } catch {
    /* storage blocked: the lesson still finishes */
  }
}

/* ── availability (pure) ──────────────────────────────────────────────── */

export type LessonAvailability =
  | { state: "ready"; sha256: string }
  | { state: "pending"; reason: string }
  | { state: "unavailable"; reason: string };

/** Can this lesson run here? Only on the exact artifact the release manifest
 *  publishes for it; never on whatever map happens to share its name. */
export function lessonAvailability(
  tour: Tour,
  status: ManifestStatus,
  datasets: readonly string[] | null,
): LessonAvailability {
  const lessonId = lessonIdOf(tour.id);
  if (status.state === "unloaded") return { state: "pending", reason: "Checking the release manifest…" };
  if (status.state === "absent")
    return { state: "unavailable", reason: "this deploy publishes no release manifest (experience.json) to check the map against." };
  if (status.state === "invalid")
    return { state: "unavailable", reason: "this deploy's release manifest (experience.json) did not validate." };
  const li = status.manifest.learn_intro;
  if (!li || li.lesson_id !== lessonId || li.dataset_id !== tour.manifest?.dataset)
    return { state: "unavailable", reason: "the release manifest does not publish a map for this lesson." };
  const art = findArtifact(status.manifest, li.dataset_id, li.sha256);
  if (!art) return { state: "unavailable", reason: "the release manifest names a map file it does not list." };
  if (datasets && !datasets.includes(li.dataset_id))
    return { state: "unavailable", reason: `the map ${li.dataset_id} is not in this deploy.` };
  return { state: "ready", sha256: li.sha256 };
}

/* ── focus hand-back ─────────────────────────────────────────────────── */

/** Set when the reader leaves a lesson for the catalog, so the catalog can
 *  put focus back on the lesson's card instead of dropping it on the page. */
let returning = false;
export function noteLessonExit(): void {
  returning = true;
}
export function takeLessonExit(): boolean {
  const r = returning;
  returning = false;
  return r;
}
