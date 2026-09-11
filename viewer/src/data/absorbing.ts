/** absorbing.ts — the Waluigi absorbing-state study, as the viewer reads it.
 *
 *  `nebulai absorbing run` plays a pinned instruct model against itself under a
 *  persona rule, judges every assistant turn with a STATED DETERMINISTIC REGEX
 *  (never a model judge), and writes `out/absorbing/<study_id>/absorbing.json`.
 *  This module is the parser and the arithmetic; `chrome/AbsorbingPanel.tsx` is
 *  the rendering. They are separate so the arithmetic can be tested without a
 *  DOM, because the arithmetic is the part that can be wrong quietly.
 *
 *  ## The three numbers, and why all three ship
 *
 *  The claim "a character that slips stays slipped" is a claim about
 *  P(violate at t+1 | violated at t). On its own that number is not evidence of
 *  anything: a model that violates 40% of the time at random produces P(1|1) =
 *  0.40 and looks like a trend. So the panel never shows it alone.
 *
 *  1. **the base rate** — how often ANY eligible turn is a violation. If the
 *     conditional's interval spans it, there is no effect to talk about.
 *  2. **the Wilson interval** on the conditional, because k/n with n = 40 is a
 *     point estimate pretending to be a measurement.
 *  3. **the within-conversation shuffle null**, because a conditional that
 *     clears the base rate can still be pure heterogeneity — some conversations
 *     violate more than others, and shuffling each conversation's OWN sequence
 *     keeps that difference while destroying the ordering. Clearing the base
 *     rate but not this null is a real, nameable outcome and has its own
 *     verdict (`above_base_rate_explained_by_heterogeneity`).
 *
 *  Nothing here recomputes the backend's verdict. The backend decided it with
 *  the full 2,000-draw null in hand; recomputing a second opinion in the viewer
 *  would give two answers to one question. What IS recomputed is the interval
 *  arithmetic — as a CHECK: `intervalDisagreement()` reports it when the
 *  shipped interval and a freshly computed Wilson interval differ, because a
 *  silent disagreement between the file and the formula is worth seeing.
 */

/** Resolve the artifact root lazily — `data/base.ts` reads `location.href` at
 *  module scope, which is correct in a browser and absent under vitest. */
async function dataBase(): Promise<string> {
  const { DATA_BASE } = await import("./base");
  return DATA_BASE;
}

/** The backend's three outcomes. `unknown` covers a document written by a
 *  version that did not rule, and is never treated as a pass. */
export type AbsorbingVerdict =
  | "absorbing_above_null"
  | "above_base_rate_explained_by_heterogeneity"
  | "not_absorbing"
  | "unknown";

export interface Rate {
  k: number;
  n: number;
  p: number;
  /** the interval as the STUDY recorded it, null when it recorded none */
  ci95: [number, number] | null;
}

export interface AbsorbingRule {
  id: string;
  statement: string;
  pattern: string;
  persona: string;
  /** how a turn was judged — printed verbatim, never paraphrased */
  judge: string;
}

export interface AbsorbingStudy {
  studyId: string;
  model: string;
  /** the resolved commit sha of the weights; never a tag */
  revision: string;
  created: string;
  elapsedS: number | null;
  rule: AbsorbingRule;
  nConversations: number;
  nConversationsRequested: number;
  nTurns: number;
  nJudgedTurns: number;
  nTransitions: number;
  /** true when a deadline stopped the run before the requested N */
  stoppedEarly: boolean;
  deadlineS: number | null;
  /** the 2x2, rows = state at t, cols = state at t+1 */
  counts: { n00: number; n01: number; n10: number; n11: number };
  baseRate: Rate & { over: string };
  pGivenViolated: Rate;
  pGivenInCharacter: Rate;
  intervalSpansBaseRate: boolean;
  null: {
    method: string;
    n: number;
    statistic: string;
    mean: number | null;
    p95: number | null;
    pValue: number | null;
    note: string;
  };
  verdict: AbsorbingVerdict;
  /** the rule-choice pilot, when the run chose its own rule. Kept because a
   *  rule picked AFTER seeing the transitions would be a different study. */
  pilot: { rates: Record<string, number>; chosen: string; target: number | null } | null;
}

const VERDICTS = new Set([
  "absorbing_above_null",
  "above_base_rate_explained_by_heterogeneity",
  "not_absorbing",
]);

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function int(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? Math.round(v) : 0;
}

function ci(v: unknown): [number, number] | null {
  if (!Array.isArray(v) || v.length !== 2) return null;
  const lo = num(v[0]);
  const hi = num(v[1]);
  return lo === null || hi === null ? null : [lo, hi];
}

function rate(v: unknown): Rate {
  const d = (v ?? {}) as Record<string, unknown>;
  return { k: int(d.k), n: int(d.n), p: num(d.p) ?? 0, ci95: ci(d.ci95) };
}

/** Parse an `absorbing.json`. Strict about identity (no study id is a throw —
 *  an unnamed study cannot be quoted), tolerant about optional blocks, and
 *  never generous: an unrecognised verdict is `unknown`, not a pass. */
export function parseStudy(doc: unknown): AbsorbingStudy {
  const d = (doc ?? {}) as Record<string, any>;
  const meta = (d.meta ?? {}) as Record<string, any>;
  const cfg = (meta.config ?? {}) as Record<string, any>;
  const s = (d.stats ?? {}) as Record<string, any>;
  const r = (d.rule ?? {}) as Record<string, any>;
  const c = (s.counts ?? {}) as Record<string, any>;
  const n = (s.null ?? {}) as Record<string, any>;

  if (!meta.study_id) throw new Error("absorbing.json has no meta.study_id");

  const base = rate(s.base_rate);
  const requested = int(cfg.n_conversations_requested) || int(s.n_conversations);
  const ran = int(cfg.n_conversations_run) || int(s.n_conversations);

  return {
    studyId: String(meta.study_id),
    model: String(meta.model ?? ""),
    revision: String(meta.revision ?? ""),
    created: String(meta.created ?? ""),
    elapsedS: num(meta.elapsed_s),
    rule: {
      id: String(r.id ?? ""),
      statement: String(r.statement ?? ""),
      pattern: String(r.pattern ?? ""),
      persona: String(r.persona ?? ""),
      judge: String(r.judge ?? ""),
    },
    nConversations: int(s.n_conversations),
    nConversationsRequested: requested,
    nTurns: int(cfg.n_turns),
    nJudgedTurns: int(s.n_turns),
    nTransitions: int(s.n_transitions),
    // `stopped_early` is the backend's own flag; the count comparison is the
    // fallback for a document written before the flag existed. Either way a
    // short run says so — a study that quietly reports the N it reached under
    // the N it asked for is the exact shape of a number that looks complete.
    stoppedEarly: cfg.stopped_early === true || (requested > 0 && ran > 0 && ran < requested),
    deadlineS: num(cfg.deadline_s),
    counts: { n00: int(c.n00), n01: int(c.n01), n10: int(c.n10), n11: int(c.n11) },
    baseRate: { ...base, over: String((s.base_rate ?? {}).over ?? "") },
    pGivenViolated: rate(s.p_violate_given_violated),
    pGivenInCharacter: rate(s.p_violate_given_in_character),
    intervalSpansBaseRate: s.interval_spans_base_rate === true,
    null: {
      method: String(n.method ?? ""),
      n: int(n.n),
      statistic: String(n.statistic ?? ""),
      mean: num(n.mean),
      p95: num(n.p95),
      pValue: num(n.p_value),
      note: String(n.note ?? ""),
    },
    verdict:
      typeof s.verdict === "string" && VERDICTS.has(s.verdict)
        ? (s.verdict as AbsorbingVerdict)
        : "unknown",
    pilot: d.pilot
      ? {
          rates: (d.pilot.rates ?? {}) as Record<string, number>,
          chosen: String(d.pilot.chosen ?? ""),
          target: num(d.pilot.target),
        }
      : null,
  };
}

// ── the arithmetic ───────────────────────────────────────────────────────────

/** 1.959963985 — the two-sided 95% normal quantile. Spelled out rather than
 *  rounded to 1.96 so the interval matches the backend's `scipy`-free formula
 *  digit for digit instead of drifting in the fourth place. */
export const Z95 = 1.959963984540054;

/** The Wilson score interval for a binomial proportion.
 *
 *  NOT the normal approximation. At the sample sizes this study produces — a
 *  conditional over the turns that FOLLOW a violation, which is a fraction of a
 *  fraction — `p ± z·sqrt(p(1-p)/n)` puts the bound below 0 or above 1 and, at
 *  p = 0 or p = 1, collapses to a zero-width interval, which reads as certainty
 *  earned from no data at all. Wilson does neither.
 *
 *  n = 0 has no interval. It returns null rather than [0, 1]: "nothing was
 *  measured" and "measured, and it could be anything" are different facts, and
 *  only one of them is honest about a missing denominator. */
export function wilson(k: number, n: number, z = Z95): [number, number] | null {
  if (!Number.isFinite(k) || !Number.isFinite(n) || n <= 0) return null;
  const kk = Math.max(0, Math.min(n, k));
  const p = kk / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/** Does an interval contain a value? The base-rate test, spelled as a function
 *  so the panel and the tests cannot disagree about the inclusive endpoints. */
export function spans(interval: [number, number] | null, value: number): boolean {
  return interval !== null && value >= interval[0] && value <= interval[1];
}

/** The 2x2 read back as rates, from the counts alone.
 *
 *  This exists to CHECK the file, not to replace it. `n01 / (n00 + n01)` is
 *  P(violate | in character) and `n11 / (n10 + n11)` is P(violate | violated);
 *  the base rate over eligible t+1 turns is `(n01 + n11)` over the whole
 *  matrix. If those disagree with the shipped numbers, one of them is wrong and
 *  the panel should say so rather than print the prettier one. */
export function ratesFromCounts(c: AbsorbingStudy["counts"]): {
  baseRate: Rate;
  pGivenViolated: Rate;
  pGivenInCharacter: Rate;
} {
  const mk = (k: number, n: number): Rate => ({
    k,
    n,
    p: n > 0 ? k / n : 0,
    ci95: wilson(k, n),
  });
  const total = c.n00 + c.n01 + c.n10 + c.n11;
  return {
    baseRate: mk(c.n01 + c.n11, total),
    pGivenViolated: mk(c.n11, c.n10 + c.n11),
    pGivenInCharacter: mk(c.n01, c.n00 + c.n01),
  };
}

/** Where the shipped numbers and the formula disagree by more than `tol`.
 *
 *  Returns one line per disagreement and an empty array when the file checks
 *  out. A disagreement is not automatically the file's fault — it is a fact
 *  about the pair — so the wording names both sides. */
export function intervalDisagreement(study: AbsorbingStudy, tol = 5e-4): string[] {
  const out: string[] = [];
  const check = (name: string, got: Rate) => {
    const want = wilson(got.k, got.n);
    if (got.ci95 === null || want === null) return;
    if (Math.abs(got.ci95[0] - want[0]) > tol || Math.abs(got.ci95[1] - want[1]) > tol) {
      out.push(
        `${name}: the study recorded [${got.ci95[0].toFixed(4)}, ${got.ci95[1].toFixed(4)}] ` +
          `for ${got.k}/${got.n}; the Wilson interval for those counts is ` +
          `[${want[0].toFixed(4)}, ${want[1].toFixed(4)}].`,
      );
    }
  };
  check("base rate", study.baseRate);
  check("P(1|1)", study.pGivenViolated);
  check("P(1|0)", study.pGivenInCharacter);

  const derived = ratesFromCounts(study.counts);
  const cmp = (name: string, got: Rate, want: Rate) => {
    if (got.n === 0 && want.n === 0) return;
    if (got.k !== want.k || got.n !== want.n) {
      out.push(
        `${name}: the study recorded ${got.k}/${got.n}; the transition matrix says ` +
          `${want.k}/${want.n}.`,
      );
    }
  };
  cmp("P(1|1)", study.pGivenViolated, derived.pGivenViolated);
  cmp("P(1|0)", study.pGivenInCharacter, derived.pGivenInCharacter);
  return out;
}

/** The one-sentence reading of a verdict. Every branch says what the result
 *  does NOT establish, because the failure modes of this measurement are the
 *  interesting part and a bare `not_absorbing` reads as "nothing happened"
 *  when it actually means "the conditional did not clear its own base rate". */
export function verdictNote(study: AbsorbingStudy): string {
  const bp = study.baseRate.p;
  const cp = study.pGivenViolated.p;
  switch (study.verdict) {
    case "absorbing_above_null":
      return (
        `After a violation the next turn violates ${(cp * 100).toFixed(1)}% of the time, ` +
        `against a base rate of ${(bp * 100).toFixed(1)}% — and the gap survives a null ` +
        `that reshuffles each conversation's own violation sequence, so it is not just ` +
        `that some conversations violate more than others. This is the measurement the ` +
        `absorbing-state claim asks for; it is a claim about this rule on this model at ` +
        `this temperature, and about nothing else.`
      );
    case "above_base_rate_explained_by_heterogeneity":
      return (
        `P(violate | violated) is ${(cp * 100).toFixed(1)}% against a ${(bp * 100).toFixed(1)}% ` +
        `base rate, but the within-conversation shuffle reaches the same number. The ` +
        `conversations differ from each other; a violation is not pulling the turn after ` +
        `it. Reporting the first sentence without the second would be the error this ` +
        `null exists to catch.`
      );
    case "not_absorbing":
      return (
        `P(violate | violated) is ${(cp * 100).toFixed(1)}% and the base rate is ` +
        `${(bp * 100).toFixed(1)}%; the conditional's interval does not clear it. This is ` +
        `NOT evidence that no absorbing state exists — it is this rule, on this model, ` +
        `at this N, failing to show one.`
      );
    default:
      return (
        `This study carries no verdict. The numbers below are still what was measured; ` +
        `what is absent is the ruling, and an absent ruling is not a negative one.`
      );
  }
}

/** Fetch one study by id. `base` is injectable so tests never touch the
 *  network and a static deploy under a sub-path still resolves. */
export async function loadStudy(studyId: string, base?: string): Promise<AbsorbingStudy> {
  const root = base ?? (await dataBase());
  const res = await fetch(`${root}/absorbing/${encodeURIComponent(studyId)}/absorbing.json`);
  if (!res.ok) throw new Error(`no absorbing study at ${studyId} (${res.status})`);
  return parseStudy(await res.json());
}

/** Every study this deploy ships, from `out/absorbing/index.json`. An absent
 *  index means no studies, not an error — the study is optional and a viewer
 *  without one still runs. */
export async function loadStudyIndex(base?: string): Promise<{ studyId: string }[]> {
  try {
    const root = base ?? (await dataBase());
    const res = await fetch(`${root}/absorbing/index.json`);
    if (!res.ok) return [];
    const doc = await res.json();
    const list = Array.isArray(doc) ? doc : (doc?.studies ?? []);
    return (Array.isArray(list) ? list : [])
      .map((s: any) => ({ studyId: String(s?.study_id ?? s?.studyId ?? s) }))
      .filter((s) => s.studyId);
  } catch {
    return [];
  }
}
