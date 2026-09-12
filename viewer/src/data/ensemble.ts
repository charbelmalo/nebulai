/** ensemble.ts — N runs of one protocol, as the viewer reads them (Attractors P3).
 *
 *  `seer run … --repeat N` runs one protocol N times under N seeds and records
 *  the membership; the server rebuilds `/seer/ensemble/<id>` from the runs' own
 *  logs on every request. This module parses that document and holds the
 *  arithmetic; `chrome/EnsemblePanel.tsx` renders it. They are separate for the
 *  same reason `absorbing.ts` and `AbsorbingPanel.tsx` are: the arithmetic is
 *  the part that can be wrong quietly, and it should be testable without a DOM.
 *
 *  ## The four rules this module exists to hold
 *
 *  **1. Below `point_estimate_min_runs` there is no point estimate.** Not a
 *  point with a caveat next to it, not a point in a lighter colour — an
 *  interval. A reader takes the number they can read off the screen, and 6/10
 *  rendered as "60%" will be quoted as 60% no matter what the footnote says.
 *  The threshold is read out of the DOCUMENT (`point_estimate_min_runs`, which
 *  the backend writes from its own constant) rather than hard-coded here, so
 *  the panel and the backend cannot drift to two different numbers.
 *
 *  **2. A fan step's `n` is part of the geometry, not a footnote.** Runs that
 *  ended early are absent from the steps they never reached, so the envelope
 *  narrows towards the right for a reason that has nothing to do with the
 *  agent agreeing with itself. `survivorship()` finds where `n` first drops
 *  below the full sample so the driver can mark that boundary, and the panel
 *  says how many runs are left at the last step.
 *
 *  **3. `missing` is never zero and never blank.** Every quantity the backend
 *  could not compute arrives with a reason attached, and those reasons are
 *  rendered. `fidelity: "missing"` on a rate means the rate does not exist;
 *  `p: null` is not `p: 0`.
 *
 *  **4. Δ̂ is reported with its terms, sign intact.** A Δ̂ near zero because two
 *  conditions are genuinely alike and a Δ̂ near zero because both are internally
 *  noisy are different findings, and only `between` / `within_a` / `within_b`
 *  tell them apart. A negative Δ̂ — conditions differing LESS than each differs
 *  from itself — is a real answer and is never clipped.
 */

import { wilson, Z95 } from "./absorbing";

/** The backend's fidelity vocabulary, as it reaches the viewer. */
export type Fidelity =
  | "native"
  | "deterministic"
  | "estimated"
  | "heuristic"
  | "missing"
  | "dropped_by_policy";

/** A rate with an interval. `p` is null exactly when `n` is 0 — no trials is
 *  not a rate of zero, and the type says so rather than a comment. */
export interface EnsembleRate {
  key: string;
  k: number;
  n: number;
  p: number | null;
  ci95: [number, number] | null;
  fidelity: Fidelity;
  /** the backend's reason, when there is no rate */
  missing: string | null;
  /** e.g. "runs with no edits cannot answer this and are excluded from n" */
  note: string | null;
  /** `verified_after_last_edit` carries this: runs that could not answer */
  nUndecidable: number | null;
  /** `completed` carries this: runs that have reached a terminal state */
  nTerminal: number | null;
}

/** One step of the fan: the per-step median and envelope over the runs that
 *  reached that step. */
export interface FanStep {
  step: number;
  median: number;
  lo: number;
  hi: number;
  /** how many runs reached this step — always ≤ the ensemble's n_runs */
  n: number;
}

export interface EnsembleRun {
  runId: string;
  index: number;
  condition: string;
  protocolId: string;
  seed: number | null;
  state: string;
  outcome: string;
  nTurns: number;
  nEvents: number;
  /** completed turns this run contributed to the fan */
  nSteps: number;
}

export interface EnsembleCondition {
  condition: string;
  protocolId: string;
  nRuns: number;
  runIds: string[];
}

export interface Reliability {
  statistic: string;
  source: string;
  formula: string | null;
  /** null when N was too small, or every run's feature vector was identical */
  deltaHat: number | null;
  between: number | null;
  withinA: number | null;
  withinB: number | null;
  bandwidth: number | null;
  kernel: string | null;
  estimator: string | null;
  nSplits: number | null;
  seed: number | null;
  nA: number | null;
  nB: number | null;
  conditionA: string | null;
  conditionB: string | null;
  featuresUsed: string[];
  /** feature name → why it was dropped from EVERY run's vector */
  featuresDropped: Record<string, string>;
  fidelity: Fidelity;
  missing: string | null;
}

export interface Ensemble {
  ensembleId: string;
  schemaVersion: number;
  created: string;
  protocol: Record<string, unknown>;
  protocolId: string | null;
  nRuns: number;
  nRunsRequested: number;
  runIds: string[];
  conditions: EnsembleCondition[];
  runs: EnsembleRun[];
  seedBase: number | null;
  seedApplied: boolean;
  fanMetric: string;
  /** the envelope's name, e.g. "p10_p90" — rendered verbatim, never as "range" */
  fanEnvelope: string;
  fanFidelity: Fidelity;
  fan: FanStep[];
  rates: EnsembleRate[];
  reliability: Reliability;
  /** the backend's own threshold for rule 1 */
  pointEstimateMinRuns: number;
  minRunsForFan: number;
  minRunsPerConditionForReliability: number;
  fidelity: Fidelity;
  /** quantity → why it could not be computed. Empty when everything computed. */
  missing: Record<string, string>;
  budget: Record<string, unknown> | null;
  /** the document as it arrived, for anything the panel wants verbatim */
  raw: unknown;
}

const FIDELITIES = new Set<string>([
  "native",
  "deterministic",
  "estimated",
  "heuristic",
  "missing",
  "dropped_by_policy",
]);

/** Anything the backend did not name is `missing`, never a pass. A document
 *  written by a newer backend that invents a fidelity we cannot interpret is
 *  a document whose claim we cannot vouch for. */
function fidelity(v: unknown): Fidelity {
  const s = typeof v === "string" ? v : "";
  return (FIDELITIES.has(s) ? s : "missing") as Fidelity;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

function interval(v: unknown): [number, number] | null {
  if (!Array.isArray(v) || v.length !== 2) return null;
  const a = num(v[0]);
  const b = num(v[1]);
  return a === null || b === null ? null : [a, b];
}

function parseRate(key: string, d: any): EnsembleRate {
  const n = Math.max(0, Math.trunc(num(d?.n) ?? 0));
  return {
    key,
    k: Math.max(0, Math.trunc(num(d?.k) ?? 0)),
    n,
    // p is taken from the document and is null when the document says null.
    // It is NOT recomputed as k/n: a document that disagrees with its own
    // counts is something to surface, not something to paper over.
    p: num(d?.p),
    ci95: interval(d?.ci95),
    fidelity: fidelity(d?.fidelity),
    missing: str(d?.missing),
    note: str(d?.note),
    nUndecidable: num(d?.n_undecidable),
    nTerminal: num(d?.n_terminal),
  };
}

function parseReliability(d: any): Reliability {
  return {
    statistic: str(d?.statistic) ?? "delta_hat",
    source: str(d?.source) ?? "",
    formula: str(d?.formula),
    deltaHat: num(d?.delta_hat),
    between: num(d?.between),
    withinA: num(d?.within_a),
    withinB: num(d?.within_b),
    bandwidth: num(d?.bandwidth),
    kernel: str(d?.kernel),
    estimator: str(d?.estimator),
    nSplits: num(d?.n_splits),
    seed: num(d?.seed),
    nA: num(d?.n_a),
    nB: num(d?.n_b),
    conditionA: str(d?.condition_a),
    conditionB: str(d?.condition_b),
    featuresUsed: Array.isArray(d?.features_used) ? d.features_used.map(String) : [],
    featuresDropped:
      d?.features_dropped && typeof d.features_dropped === "object"
        ? Object.fromEntries(
            Object.entries(d.features_dropped as Record<string, unknown>).map(([k, v]) => [
              k,
              String(v),
            ]),
          )
        : {},
    fidelity: fidelity(d?.fidelity),
    missing: str(d?.missing),
  };
}

export function parseEnsemble(doc: any): Ensemble {
  const id = str(doc?.ensemble_id);
  if (!id) {
    // An ensemble with no id is not an ensemble with a blank name — it is a
    // document we cannot attribute to a run of anything.
    throw new Error("ensemble document has no ensemble_id");
  }
  const rates = Object.entries((doc?.rates ?? {}) as Record<string, any>).map(([k, v]) =>
    parseRate(k, v),
  );
  const fan: FanStep[] = (Array.isArray(doc?.fan) ? doc.fan : [])
    .map((s: any) => ({
      step: Math.trunc(num(s?.step) ?? 0),
      median: num(s?.median) ?? 0,
      lo: num(s?.lo) ?? 0,
      hi: num(s?.hi) ?? 0,
      n: Math.max(0, Math.trunc(num(s?.n) ?? 0)),
    }))
    .sort((a: FanStep, b: FanStep) => a.step - b.step);

  return {
    ensembleId: id,
    schemaVersion: Math.trunc(num(doc?.schema_version) ?? 0),
    created: str(doc?.created) ?? "",
    protocol: (doc?.protocol ?? {}) as Record<string, unknown>,
    protocolId: str(doc?.protocol?.id),
    nRuns: Math.max(0, Math.trunc(num(doc?.n_runs) ?? 0)),
    nRunsRequested: Math.max(0, Math.trunc(num(doc?.n_runs_requested) ?? 0)),
    runIds: Array.isArray(doc?.run_ids) ? doc.run_ids.map(String) : [],
    conditions: (Array.isArray(doc?.conditions) ? doc.conditions : []).map((c: any) => ({
      condition: String(c?.condition ?? ""),
      protocolId: String(c?.protocol_id ?? ""),
      nRuns: Math.max(0, Math.trunc(num(c?.n_runs) ?? 0)),
      runIds: Array.isArray(c?.run_ids) ? c.run_ids.map(String) : [],
    })),
    runs: (Array.isArray(doc?.runs) ? doc.runs : []).map((r: any) => ({
      runId: String(r?.run_id ?? ""),
      index: Math.trunc(num(r?.index) ?? 0),
      condition: String(r?.condition ?? ""),
      protocolId: String(r?.protocol_id ?? ""),
      seed: num(r?.seed),
      state: String(r?.state ?? "unknown"),
      outcome: String(r?.outcome ?? "unknown"),
      nTurns: Math.max(0, Math.trunc(num(r?.n_turns) ?? 0)),
      nEvents: Math.max(0, Math.trunc(num(r?.n_events) ?? 0)),
      nSteps: Math.max(0, Math.trunc(num(r?.n_steps) ?? 0)),
    })),
    seedBase: num(doc?.seed_base),
    seedApplied: doc?.seed_applied === true,
    fanMetric: str(doc?.fan_metric) ?? "unknown",
    fanEnvelope: str(doc?.fan_envelope) ?? "unknown",
    fanFidelity: fidelity(doc?.fan_fidelity),
    fan,
    rates,
    reliability: parseReliability(doc?.reliability),
    // A document that does not state the threshold gets the strictest reading
    // available: Infinity, which means "no point estimate is ever allowed".
    // Defaulting to 20 would silently invent the backend's constant, and
    // defaulting to 0 would license every point estimate in a document that
    // never agreed to any.
    pointEstimateMinRuns: num(doc?.point_estimate_min_runs) ?? Number.POSITIVE_INFINITY,
    minRunsForFan: num(doc?.min_runs_for_fan) ?? 3,
    minRunsPerConditionForReliability: num(doc?.min_runs_per_condition_for_reliability) ?? 4,
    fidelity: fidelity(doc?.fidelity),
    missing:
      doc?.missing && typeof doc.missing === "object"
        ? Object.fromEntries(
            Object.entries(doc.missing as Record<string, unknown>).map(([k, v]) => [k, String(v)]),
          )
        : {},
    budget: (doc?.budget ?? null) as Record<string, unknown> | null,
    raw: doc,
  };
}

/* ── rule 1: the point estimate ─────────────────────────────────────────── */

/** May a rate be shown as a single number?
 *
 *  Only when the ensemble has at least the backend's stated minimum AND this
 *  particular rate's own denominator does too. `verified_after_last_edit`
 *  excludes runs that could not answer, so an ensemble of 24 runs can carry a
 *  rate with n = 5 — and that rate is as under-powered as a 5-run ensemble. */
export function pointEstimateAllowed(e: Ensemble, r: EnsembleRate): boolean {
  if (r.n <= 0 || r.p === null) return false;
  return e.nRuns >= e.pointEstimateMinRuns && r.n >= e.pointEstimateMinRuns;
}

/** The interval to draw for a rate: the document's, or a recomputed Wilson when
 *  the document carried none. Null when there is nothing to draw. */
export function rateInterval(r: EnsembleRate): [number, number] | null {
  if (r.n <= 0) return null;
  return r.ci95 ?? wilson(r.k, r.n);
}

/** The same CHECK `absorbing.ts` runs: recompute every shipped interval and
 *  report the ones that disagree, rather than quietly preferring one. */
export function intervalDisagreement(e: Ensemble, tol = 5e-4): string[] {
  const out: string[] = [];
  for (const r of e.rates) {
    if (!r.ci95 || r.n <= 0) continue;
    const mine = wilson(r.k, r.n);
    if (!mine) continue;
    if (Math.abs(mine[0] - r.ci95[0]) > tol || Math.abs(mine[1] - r.ci95[1]) > tol) {
      out.push(
        `${r.key}: the file says [${r.ci95[0].toFixed(4)}, ${r.ci95[1].toFixed(4)}] ` +
          `but Wilson on ${r.k}/${r.n} gives [${mine[0].toFixed(4)}, ${mine[1].toFixed(4)}]`,
      );
    }
    if (r.p !== null && r.n > 0 && Math.abs(r.p - r.k / r.n) > tol) {
      out.push(
        `${r.key}: the file's rate is ${r.p.toFixed(4)} but its own counts say ` +
          `${r.k}/${r.n} = ${(r.k / r.n).toFixed(4)}`,
      );
    }
  }
  return out;
}

/* ── rule 2: survivorship in the fan ────────────────────────────────────── */

export interface Survivorship {
  /** the largest per-step n anywhere in the fan */
  nAtStart: number;
  /** the per-step n at the last step */
  nAtEnd: number;
  /** the first step whose n is below `nAtStart`, or null when none is */
  firstDropStep: number | null;
  /** true when the fan narrows partly because runs left it */
  narrowsByAttrition: boolean;
}

export function survivorship(e: Ensemble): Survivorship {
  if (!e.fan.length) {
    return { nAtStart: 0, nAtEnd: 0, firstDropStep: null, narrowsByAttrition: false };
  }
  const nAtStart = Math.max(...e.fan.map((s) => s.n));
  const nAtEnd = e.fan[e.fan.length - 1]!.n;
  const drop = e.fan.find((s) => s.n < nAtStart);
  return {
    nAtStart,
    nAtEnd,
    firstDropStep: drop ? drop.step : null,
    narrowsByAttrition: nAtEnd < nAtStart,
  };
}

/** The fan's envelope width per step — what the reader is actually meant to
 *  read. Exposed so the panel can say "the band is this wide" in the same
 *  units as the metric, instead of leaving it to the eye. */
export function envelopeWidths(e: Ensemble): { step: number; width: number; n: number }[] {
  return e.fan.map((s) => ({ step: s.step, width: s.hi - s.lo, n: s.n }));
}

/** Do two ensembles' envelopes overlap at every shared step?
 *
 *  If they do, their medians have not been distinguished by this data, whatever
 *  the medians say. This returns the shared steps where the bands are DISJOINT
 *  — an empty result means "no step separates them", which is the honest
 *  headline for two fans that look different to the eye. */
export function disjointSteps(a: Ensemble, b: Ensemble): number[] {
  const byStep = new Map(b.fan.map((s) => [s.step, s]));
  const out: number[] = [];
  for (const s of a.fan) {
    const t = byStep.get(s.step);
    if (!t) continue;
    if (s.hi < t.lo || t.hi < s.lo) out.push(s.step);
  }
  return out;
}

/* ── rule 4: reading Δ̂ ──────────────────────────────────────────────────── */

/** A sentence about Δ̂ that stays inside what Δ̂ supports.
 *
 *  Deliberately not a verdict function. §6.4.1 gives no threshold at which Δ̂
 *  becomes "significant", and inventing one here — Δ̂ > 0.1 is a difference! —
 *  would be the viewer legislating statistics the backend never claimed. What
 *  this does instead is name which of the three terms is carrying the answer,
 *  which is the thing a reader cannot see from the single number. */
export function reliabilityNote(r: Reliability): string {
  if (r.deltaHat === null) {
    return (
      r.missing ??
      "Δ̂ was not computed, and the document does not say why — treat the two " +
        "conditions as undistinguished."
    );
  }
  const within = ((r.withinA ?? 0) + (r.withinB ?? 0)) / 2;
  const between = r.between ?? 0;
  const d = r.deltaHat;
  const terms =
    `between = ${between.toFixed(4)}, within = ${within.toFixed(4)} ` +
    `(A ${(r.withinA ?? 0).toFixed(4)}, B ${(r.withinB ?? 0).toFixed(4)})`;

  if (d < 0) {
    return (
      `Δ̂ = ${d.toFixed(4)} is NEGATIVE: ${r.conditionA ?? "A"} and ` +
      `${r.conditionB ?? "B"} differ from each other LESS than each differs from ` +
      `itself. That is a real answer, not a failed measurement, and it is not ` +
      `clipped to zero — it says the contrast you are looking at is smaller than ` +
      `this protocol's own run-to-run noise. ${terms}.`
    );
  }
  if (within > 0 && between <= within) {
    return (
      `Δ̂ = ${d.toFixed(4)}. The between-condition separation does not exceed the ` +
      `average within-condition separation, so whatever the medians look like, ` +
      `these two conditions have not been told apart by ${r.nA ?? "?"} and ` +
      `${r.nB ?? "?"} runs. ${terms}.`
    );
  }
  return (
    `Δ̂ = ${d.toFixed(4)}: the conditions separate by more than each condition's ` +
    `own internal spread, over ${r.nSplits ?? "?"} split-half draws at seed ` +
    `${r.seed ?? "?"}. This is a measurement of the two conditions AS RUN — same ` +
    `harness, same seeds, these features only — and says nothing about either ` +
    `condition on its own. ${terms}.`
  );
}

/** The sentence the panel puts above a rate that is too small for a point.
 *  One place, so the wording cannot drift between call sites. */
export function underpoweredNote(e: Ensemble, r: EnsembleRate): string {
  const min = Number.isFinite(e.pointEstimateMinRuns)
    ? `${e.pointEstimateMinRuns}`
    : "the threshold this document does not state";
  return (
    `${r.k}/${r.n} — shown as an interval, not as a percentage. This ensemble ` +
    `has ${e.nRuns} run${e.nRuns === 1 ? "" : "s"} and a point estimate needs ${min}.`
  );
}

/** Everything the document could not compute, as lines to render. Includes the
 *  per-rate reasons, which live on the rates rather than in `missing`. */
export function missingLines(e: Ensemble): string[] {
  const out = Object.entries(e.missing).map(([k, v]) => `${k}: ${v}`);
  for (const r of e.rates) {
    if (r.fidelity === "missing" && r.missing) out.push(`${r.key}: ${r.missing}`);
  }
  return out;
}

/** Did this ensemble run as many times as it was asked to? */
export function shortfall(e: Ensemble): number {
  return Math.max(0, e.nRunsRequested - e.nRuns);
}

export { wilson, Z95 };
