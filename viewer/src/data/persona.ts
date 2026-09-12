/** persona.ts — the fixed coordinate system a trajectory is drawn in (D4).
 *
 *  `out/persona/<space_id>/space.json` is written once by `nebulai persona
 *  build` and never refitted: the basis, the frozen prompt set's sha256, and
 *  the label-permutation control all travel together. Placing a run is a pure
 *  projection through a basis that already exists, which is the only reason a
 *  trajectory drawn today is comparable with one drawn last month.
 *
 *  The one rule this module exists to enforce, from §3.3 of the plan:
 *
 *  > a space whose PC1 does not clear its permutation null renders with
 *  > `verdict` visible and **cannot be used as the default trajectory
 *  > coordinate system**.
 *
 *  `isSelectableAsDefault()` is that rule, in one place, as a function of the
 *  verdict alone. A caller that wants to look at a failed space anyway can —
 *  `load()` returns it, the archetype scatter draws, and `verdictNote()` says
 *  in words what the number means. What it cannot do is become the default
 *  quietly, because then risk A (a persona axis that is really a prompt-length
 *  axis) would be invisible rather than survivable.
 *
 *  Nothing here invents a coordinate. A space.json that is missing its control
 *  block is `unknown`, not `above_null`, and `unknown` is not selectable
 *  either — an unmeasured control and a failed one are different facts, and
 *  `verdictNote` keeps them apart, but neither is a pass.
 */

/** Resolve the artifact root lazily.
 *
 *  `data/base.ts` reads `location.href` at module scope, which is correct in a
 *  browser and absent in a unit-test runner. A static import would therefore
 *  make the pure half of this module — the parser and the selectability rule,
 *  which are the parts that most need testing — unloadable outside a DOM. */
async function dataBase(): Promise<string> {
  const { DATA_BASE } = await import("./base");
  return DATA_BASE;
}

/** The control's three outcomes, plus the state of a space that never ran one. */
export type PersonaVerdict = "above_null" | "at_null" | "below_null" | "unknown";

export interface PersonaControl {
  method: string;
  n: number;
  pc1Evr: number | null;
  pc1EvrNullMean: number | null;
  pc1EvrNullP95: number | null;
  pValue: number | null;
  verdict: PersonaVerdict;
  /** A control that was measured, failed, and was kept in the artifact. Drawn
   *  beside the headline one rather than dropped, because a null that was run
   *  first and rejected is evidence about the space, not an embarrassment. */
  crossCheck: {
    method: string;
    verdict: PersonaVerdict;
    pc1EvrNullP95: number | null;
    note: string;
  } | null;
}

export interface PersonaArchetype {
  name: string;
  /** Scores on each component, in basis order. `scores[0]` is PC1. */
  scores: number[];
}

export interface PersonaSpace {
  spaceId: string;
  model: string;
  /** The resolved commit sha of the weights. Never a tag, never a guess. */
  revision: string;
  layer: number;
  pooling: string;
  promptSet: { id: string; sha256: string; n: number };
  created: string;
  /** Explained-variance ratio per component, over TOTAL variance. */
  evr: number[];
  archetypes: PersonaArchetype[];
  control: PersonaControl;
}

const VERDICTS = new Set(["above_null", "at_null", "below_null"]);

function asVerdict(v: unknown): PersonaVerdict {
  return typeof v === "string" && VERDICTS.has(v) ? (v as PersonaVerdict) : "unknown";
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Parse a `space.json` document. Tolerant about what is absent and strict
 *  about what absence means: a missing control is `unknown`, never a pass. */
export function parseSpace(doc: unknown): PersonaSpace {
  const d = (doc ?? {}) as Record<string, any>;
  const meta = (d.meta ?? {}) as Record<string, any>;
  const basis = (d.basis ?? {}) as Record<string, any>;
  const control = (d.control ?? {}) as Record<string, any>;
  const cross = control.cross_check as Record<string, any> | undefined;
  const prompt = (meta.prompt_set ?? {}) as Record<string, any>;

  if (!meta.space_id) throw new Error("space.json has no meta.space_id");

  return {
    spaceId: String(meta.space_id),
    model: String(meta.model ?? ""),
    revision: String(meta.revision ?? ""),
    layer: Number(meta.layer ?? -1),
    pooling: String(meta.pooling ?? ""),
    promptSet: {
      id: String(prompt.id ?? ""),
      sha256: String(prompt.sha256 ?? ""),
      n: Number(prompt.n ?? 0),
    },
    created: String(meta.created ?? ""),
    evr: Array.isArray(basis.explained_variance_ratio)
      ? basis.explained_variance_ratio.map((x: unknown) => Number(x) || 0)
      : [],
    archetypes: Array.isArray(d.archetypes)
      ? d.archetypes.map((a: any) => ({
          name: String(a?.name ?? ""),
          scores: Array.isArray(a?.scores) ? a.scores.map((x: unknown) => Number(x) || 0) : [],
        }))
      : [],
    control: {
      method: String(control.method ?? ""),
      n: Number(control.n ?? 0),
      pc1Evr: num(control.pc1_evr),
      pc1EvrNullMean: num(control.pc1_evr_null_mean),
      pc1EvrNullP95: num(control.pc1_evr_null_p95),
      pValue: num(control.p_value),
      verdict: asVerdict(control.verdict),
      crossCheck: cross
        ? {
            method: String(cross.method ?? ""),
            verdict: asVerdict(cross.verdict),
            pc1EvrNullP95: num(cross.pc1_evr_null_p95),
            note: String(cross.note ?? ""),
          }
        : null,
    },
  };
}

/** §3.3's rule, as a function. The ONLY place the default-coordinate-system
 *  decision is made; a caller that reimplements this comparison has moved the
 *  rule somewhere it can drift. */
export function isSelectableAsDefault(space: PersonaSpace): boolean {
  return space.control.verdict === "above_null";
}

/** What the verdict means, in words, for the card that always renders beside
 *  the space (R5: the null ships with the figure). */
export function verdictNote(space: PersonaSpace): string {
  const c = space.control;
  const evr = c.pc1Evr;
  const p95 = c.pc1EvrNullP95;
  const pair =
    evr !== null && p95 !== null
      ? ` PC1 explains ${(evr * 100).toFixed(1)}% of the variance; the label-permutation null's 95th percentile is ${(p95 * 100).toFixed(1)}%.`
      : "";
  switch (c.verdict) {
    case "above_null":
      return (
        `PC1 clears its permutation null.${pair} The axis is carrying something the ` +
        `labels have to be right for — it is not an artefact of the prompt set's shape.`
      );
    case "at_null":
      return (
        `PC1 does not separate from its permutation null.${pair} Shuffling the archetype ` +
        `labels produces about the same leading component, so this space cannot be the ` +
        `default coordinate system. It is still viewable; nothing drawn in it is a claim ` +
        `about personas.`
      );
    case "below_null":
      return (
        `PC1 falls BELOW its permutation null.${pair} The shuffled labels do better than ` +
        `the real ones, which means the leading component is tracking something the ` +
        `archetype labels cut across — prompt length or probe mix are the usual ` +
        `suspects. Not selectable as the default coordinate system.`
      );
    default:
      return (
        `This space carries no permutation control. That is not the same as failing one, ` +
        `and it is not a pass either: until the control has been run, a trajectory drawn ` +
        `here cannot be told apart from one drawn on a prompt-length axis.`
      );
  }
}

/** Short label for a chip beside the space name. */
export function verdictLabel(v: PersonaVerdict): string {
  switch (v) {
    case "above_null":
      return "clears null";
    case "at_null":
      return "at null";
    case "below_null":
      return "below null";
    default:
      return "no control";
  }
}

/** Fetch one space by id. `base` is injectable so the tests never touch the
 *  network and a static deploy under a sub-path still resolves. */
export async function loadSpace(spaceId: string, base?: string): Promise<PersonaSpace> {
  const root = base ?? (await dataBase());
  const res = await fetch(`${root}/persona/${encodeURIComponent(spaceId)}/space.json`);
  if (!res.ok) throw new Error(`no persona space at ${spaceId} (${res.status})`);
  return parseSpace(await res.json());
}

/** Every built space, newest first. Absent index → no spaces, not an error:
 *  the persona pipeline is optional and a viewer without one still runs. */
export async function loadSpaceIndex(base?: string): Promise<{ spaceId: string }[]> {
  try {
    const root = base ?? (await dataBase());
    const res = await fetch(`${root}/persona/index.json`);
    if (!res.ok) return [];
    const doc = await res.json();
    const list = Array.isArray(doc) ? doc : (doc?.spaces ?? []);
    return (Array.isArray(list) ? list : [])
      .map((s: any) => ({ spaceId: String(s?.space_id ?? s?.spaceId ?? s) }))
      .filter((s) => s.spaceId);
  } catch {
    return [];
  }
}

/** The 2-D scatter the persona card draws: PC1 × PC2 of every archetype,
 *  normalised into [-1, 1] against the larger of the two extents so the aspect
 *  ratio of the space is preserved rather than stretched to the box. */
export function archetypeScatter(
  space: PersonaSpace,
  a = 0,
  b = 1,
): { name: string; x: number; y: number }[] {
  const pts = space.archetypes
    .filter((p) => p.scores.length > Math.max(a, b))
    .map((p) => ({ name: p.name, x: p.scores[a]!, y: p.scores[b]! }));
  if (!pts.length) return [];
  const span = pts.reduce((m, p) => Math.max(m, Math.abs(p.x), Math.abs(p.y)), 0) || 1;
  return pts.map((p) => ({ name: p.name, x: p.x / span, y: p.y / span }));
}
