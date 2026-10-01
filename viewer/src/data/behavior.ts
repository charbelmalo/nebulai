/** Behavioral divergence data (`out/behavior/behavior.json`).
 *
 *  The artifact is produced by `nebulai behavior export`; its shape is fixed by
 *  `src/nebulai/behavior/export.py` and the data contract in
 *  docs/BEHAVIORAL-DIVERGENCE-PLAN.md §9.5. Two properties of that contract
 *  drive every type here and must survive any edit:
 *
 *  **`null` is not `0`.** A metric the pipeline could not compute is written as
 *  `null`, and the page renders it as "not measured" with the reason, never as
 *  a zero bar. So every numeric field that can go missing is typed
 *  `number | null` rather than `number`, and the accessors below refuse to
 *  coalesce. `??  0` anywhere in this file or its consumers is a bug.
 *
 *  **Position is lexical, not behavioural.** `landscape.coords` places a cue by
 *  where its own WORDS sit in a pinned neutral encoder — not by anything either
 *  model did. The caption for that is `landscape.projection.quantity_label`,
 *  computed from the fit, and the page must print it from here rather than
 *  writing its own copy; a hand-written "these axes capture most of the
 *  variance" would drift from the number the moment the cue set grows.
 *
 *  At a few hundred cues the file is small, so this is a plain fetch with no
 *  worker, following `data/compare.ts`. */

import { behaviorStudyPublished } from "./experience";
import { DATA_BASE } from "./base";

/** One arm (one pinned model deployment) of one cue. */
export interface BehaviorArm {
  model_key: string;
  n_attempted: number;
  n_valid: number;
  parse_rate: number;
  distinct_types: number;
  entropy: number;
  type_token_ratio: number;
  /** split-half reliability; null when there were too few valid trials */
  reliability: number | null;
  top_associates: string[];
  detectors: {
    cue_echo: number;
    exemplar_echo: number;
    within_trial_duplicate: number;
    prompt_copy: number;
  };
  oov_fragment_ratio: number | null;
}

/** A cue's status decides what the page is allowed to draw for it. `measured`
 *  is the only one that gets an effect mark; `gated` failed a precondition
 *  (compliance parity, informativeness floor) and `missing` was never
 *  measurable. `reasons` is never empty for the latter two — that string is
 *  what the inspector shows instead of a number. */
export type CueStatus = "measured" | "gated" | "missing";

export interface BehaviorCue {
  cue: string;
  stratum: string;
  pack: string;
  /** "R" = replication arm, "G" = generalization arm (plan §5.2) */
  partition: string;
  status: CueStatus;
  reasons: string[];
  delta_hat: number | null;
  mmd2_between: number | null;
  within: Record<string, number | null>;
  p_value: number | null;
  q_value: number | null;
  /** [lo, hi]; either end may be null */
  ci: [number | null, number | null];
  /** Manski bounds under non-response; [lo, hi] */
  manski: [number | null, number | null];
  jsd: number | null;
  jsd_n: number | null;
  rbo: number | null;
  location: number | null;
  dispersion: Record<string, number | null>;
  /** "location" | "dispersion" | "mixed" | null */
  dominant: string | null;
  /** the GPT-2-small vs GPT-2-XL capability-control Δ̂ for this cue (§5.7):
   *  the yardstick that says how big a KNOWN capability gap looks */
  capability_reference: number | null;
  arms: Record<string, BehaviorArm>;
}

export interface BehaviorLandscape {
  method: string;
  dims: number;
  /** `"missing"` when the exporter refused to fit: a study with fewer
   *  comparable cues than axes has no landscape, and an empty `coords` on its
   *  own cannot be told apart from a filter that matched nothing. Older
   *  artifacts predate the field, so absent reads as fitted. */
  status?: "measured" | "missing";
  /** why no landscape was fitted; present only with `status: "missing"` */
  reason?: string;
  coords: number[][];
  pca_mean: number[];
  /** axis-major flat: axis j is pca_axes[j*d …]; shape is [n_axes, d] */
  pca_axes: number[];
  pca_axes_shape: [number, number];
  explained_variance_ratio: number[];
  projection: {
    /** null when unmeasured — never 0, which would read as "no variance" */
    quantity: number | null;
    quantity_label: string;
    total_variance: number | null;
    encoder: string;
    encoder_revision: string;
    warning: string;
  };
  /** standard trustworthiness of the 2-D projection; null when n is too small */
  trustworthiness?: number | null;
}

export interface BehaviorModelRef {
  key: string;
  /** the PINNED deployment id — never substituted, never a family name */
  model_id: string;
  provider: string;
  revision?: string | null;
}

export interface BehaviorManifest {
  hash: string;
  protocol_hash: string;
  frozen_at: string;
  strict: boolean;
  models: BehaviorModelRef[];
  frames: { id: string; role: string }[];
  trials_per_cue: number;
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  embedder: {
    id: string;
    sha: string;
    dtype: string;
    pooling: string;
    normalize: boolean;
    secondary_id: string | null;
    second_embedder_resolution: string;
    agreement_claim: string;
  };
  statistics: {
    delta_hat_form: string;
    mmd_bandwidth: string | number;
    n_permutations: number;
    n_bootstrap: number;
    effect_floor: number;
    q_threshold: number;
    multiple_testing: string;
    rbo_p: number;
    jsd_correction: string;
    compliance_parity_max: number;
  };
  /** null when unmeasured. NEVER 0 — an arm with no key was never observed. */
  reasoning_tokens_p95: number | null;
  fingerprint_available: boolean | null;
  p_floor: number | null;
  git_commit: string | null;
}

export interface BehaviorRun {
  run_id: string;
  /** which arm this run collected (the exporter writes one record per arm) */
  arm?: string;
  started: string;
  finished: string | null;
  n_trials: number;
  n_completed: number;
  cost_usd: number | null;
  halted: string | null;
  not_run: Record<string, string>;
}

/** Stamped by `nebulai behavior publish`. `published_as: "example"` means the
 *  study was collected from a source that cannot support a claim about any
 *  model — a synthetic arm, or the hash stand-in for the encoder — and was
 *  published with `--force` to exercise the page. Every cue in such a study is
 *  already downgraded upstream; this field is what lets the page SAY so
 *  instead of merely showing statuses that look cautious for no visible
 *  reason. */
export interface BehaviorPublished {
  study_id: string;
  source: string;
  at: string;
  published_as: "study" | "example";
}

/** How much of the preregistered cue set this artifact actually covers.
 *
 *  The denominator is the load-bearing field. A reader looking at forty cues in
 *  a hundred-cue study has no way to tell a deliberately shortened run from a
 *  truncated database, and "complete" is the assumption they will make by
 *  default — which is the wrong one for anything that can be interrupted.
 *  Optional only because an artifact written before the field existed has to
 *  keep loading; `coverageNote` treats its absence as "unknown", never as
 *  "complete". */
export interface BehaviorCoverage {
  cues_planned: number;
  cues_analyzed: number;
  complete: boolean;
  reason: string;
  cue_limit?: number | null;
}

export interface BehaviorData {
  schema: string;
  generated: string;
  study_id: string;
  manifest: BehaviorManifest;
  claim: string;
  landscape: BehaviorLandscape;
  cues: BehaviorCue[];
  cue_index: Record<string, number>;
  /** absent in artifacts written before coverage was recorded */
  coverage?: BehaviorCoverage;
  diagnostics: Record<string, unknown>;
  runs: BehaviorRun[];
  samples: Record<string, { cue: string; model_key: string; text: string }[]>;
  /** absent for an artifact read straight out of a study directory */
  published?: BehaviorPublished;
}

/** True when the page must say, in words, that what it is showing is not
 *  evidence about any model. Absent metadata means an older artifact, and an
 *  older artifact is not assumed innocent: `strict_source: false` alone is
 *  enough. */
export function isExampleOnly(d: BehaviorData | null): boolean {
  if (!d) return false;
  if (d.published?.published_as === "example") return true;
  return d.diagnostics?.strict_source === false;
}

/** The sentence the page owes a reader when the study did not cover its own
 *  cue list, or `null` when it did.
 *
 *  Returns a note for an artifact with no `coverage` block too. Absence is not
 *  evidence of completeness: the field was added after the first studies were
 *  exported, and an old artifact is exactly the one whose coverage nobody can
 *  reconstruct from the cue list alone. */
export function coverageNote(d: BehaviorData | null): string | null {
  if (!d) return null;
  const c = d.coverage;
  if (!c) {
    return (
      "This artifact does not record how much of its cue list it covered. It " +
      "was exported before coverage was tracked, so the " +
      `${d.cues.length} cues below cannot be read as the whole study.`
    );
  }
  if (c.complete) return null;
  const head = `${c.cues_analyzed} of ${c.cues_planned} preregistered cues are analyzed here.`;
  return c.reason ? `${head} ${c.reason}` : head;
}

let cached: BehaviorData | null | undefined;

/** `null` = no study exported yet. That is a first-class state, not an error:
 *  a static deploy without `out/behavior/` is the normal case, and the page
 *  renders an explanation of what would produce the file. */
export async function loadBehavior(base = DATA_BASE): Promise<BehaviorData | null> {
  if (cached !== undefined) return cached;
  // the release manifest says no study ships here: do not request a file
  // the host is known not to have (it only produced a 404 in the console)
  if (behaviorStudyPublished() === false) return (cached = null);
  try {
    const res = await fetch(`${base}/behavior/behavior.json`);
    cached = res.ok ? ((await res.json()) as BehaviorData) : null;
  } catch {
    cached = null;
  }
  return cached;
}

/** Test seam. Production never calls this. */
export function __resetBehaviorCache(): void {
  cached = undefined;
}

// ---------------------------------------------------------------------------
// derived quantities the page needs — kept here so the component stays a view
// ---------------------------------------------------------------------------

/** Effect is encoded by AREA, never by radius (§8.2).
 *
 *  Radius-encoded magnitude is the classic bubble-chart lie: doubling r
 *  quadruples the ink, so a reader comparing two marks over-reads the larger
 *  effect by the square. Taking the square root of the normalised effect makes
 *  the drawn area proportional to Δ̂, which is the quantity being claimed.
 *
 *  Returns `null` for a cue with no effect — the caller draws the
 *  `indeterminate` mark, not a dot of size zero, because a zero-radius dot
 *  reads as "no effect" when the truth is "no measurement". */
export function cueMarkRadius(
  cue: BehaviorCue,
  maxEffect: number,
  minPx = 2.5,
  maxPx = 16,
): number | null {
  if (cue.status !== "measured" || cue.delta_hat === null) return null;
  if (!(maxEffect > 0)) return minPx;
  const frac = Math.min(1, Math.max(0, Math.abs(cue.delta_hat) / maxEffect));
  return minPx + (maxPx - minPx) * Math.sqrt(frac);
}

/** The largest measured |Δ̂| in the study — the scale every mark is drawn
 *  against. Cues that are gated or missing contribute nothing: letting a
 *  missing cue set the scale would shrink every real mark. */
export function maxMeasuredEffect(cues: BehaviorCue[]): number {
  let m = 0;
  for (const c of cues) {
    if (c.status === "measured" && c.delta_hat !== null) m = Math.max(m, Math.abs(c.delta_hat));
  }
  return m;
}

/** Significance under the study's own BY threshold, three-valued.
 *
 *  `null` (indeterminate) is a real answer and gets its own channel in the UI.
 *  A cue whose q could not be computed is NOT "not significant" — collapsing
 *  those two is the single most common way a screen of this kind lies. */
export function cueSignificant(cue: BehaviorCue, qThreshold: number): boolean | null {
  if (cue.status !== "measured") return null;
  if (cue.q_value === null) return null;
  return cue.q_value <= qThreshold;
}

/** Format a possibly-missing metric. The fallback is the word, never a number.
 *  Callers must pass the whole cue's reason list so the tooltip can say WHY. */
export function fmtMetric(v: number | null, digits = 3): string {
  return v === null ? "not measured" : v.toFixed(digits);
}

/** Does this study contain any arm that was not run, and why? Surfaced at the
 *  top of the page rather than buried in provenance: a two-arm study where one
 *  arm never ran is a different object from a two-arm study. */
export function notRunArms(data: BehaviorData): { key: string; reason: string }[] {
  const out = new Map<string, string>();
  for (const r of data.runs) {
    for (const [k, why] of Object.entries(r.not_run ?? {})) out.set(k, why);
  }
  return [...out].map(([key, reason]) => ({ key, reason }));
}

/** Case- and punctuation-insensitive cue search. Matches the cue itself, its
 *  stratum, its pack and any arm's top associates, so "daddy" finds the cue
 *  whether the reader remembers it as a cue or as something a model said. */
export function searchCues(cues: BehaviorCue[], query: string): BehaviorCue[] {
  const q = query.trim().toLowerCase();
  if (!q) return cues;
  return cues.filter((c) => {
    if (c.cue.toLowerCase().includes(q)) return true;
    if (c.stratum.toLowerCase().includes(q)) return true;
    if (c.pack.toLowerCase().includes(q)) return true;
    for (const arm of Object.values(c.arms)) {
      if (arm.top_associates.some((a) => a.toLowerCase().includes(q))) return true;
    }
    return false;
  });
}

/** Project a new point into the study's FIXED landscape.
 *
 *  `(x - mean) @ axes`, with `axes` unpacked from the axis-major flat layout.
 *  This is the whole reason the transform is exported: adding cues must place
 *  them in the published space rather than refitting one, which would move
 *  every cue already permalinked. */
export function projectIntoLandscape(l: BehaviorLandscape, vec: number[]): number[] {
  const [nAxes, d] = l.pca_axes_shape;
  if (l.status === "missing" || nAxes === 0 || d === 0) {
    //  There is no transform to project through. Returning [0, 0] would put the
    //  new cue at a coordinate it never earned, and at the origin of a plot that
    //  does not exist.
    throw new Error(
      `projectIntoLandscape: this study has no fitted landscape${l.reason ? ` (${l.reason})` : ""}`,
    );
  }
  if (vec.length !== d) {
    throw new Error(
      `projectIntoLandscape: vector has ${vec.length} dims, the landscape was fit on ${d}`,
    );
  }
  const out: number[] = [];
  for (let j = 0; j < nAxes; j++) {
    let acc = 0;
    for (let i = 0; i < d; i++) acc += (vec[i]! - l.pca_mean[i]!) * l.pca_axes[j * d + i]!;
    out.push(acc);
  }
  return out;
}
