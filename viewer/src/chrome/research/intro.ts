/** Research's first analysis: "Position Patterns by Frequency" read from the
 *  exact bundle the release manifest pins (`research_intro`). This module is
 *  the logic only — verification, the task's questions, the value table and
 *  progress — so it is unit-testable without a renderer.
 *
 *  The task orients a researcher; it is not a new scientific claim. It asks
 *  what the chart measures, has them read one frequency/power value, asks what
 *  the chart cannot show, and offers the exact published bytes as the export. */

import { DATA_BASE } from "../../data/base";
import type { ExperienceManifest } from "../../data/experience";
import type { FourierBundle } from "../../data/interp";

export type ResearchIntro = ExperienceManifest["research_intro"];

/* ── verification ─────────────────────────────────────────────────────── */

export type BundleCheck =
  | { state: "pending" }
  | { state: "verified"; bytes: Uint8Array; json: FourierBundle; url: string }
  /** the file is not on this server */
  | { state: "missing"; url: string; status: number | null; message: string }
  /** a file is there, but it is not the pinned one */
  | { state: "mismatch"; url: string; expected: string; actual: string; bytes: number };

export function bundleUrl(ri: ResearchIntro, base = DATA_BASE): string {
  return `${base}/${ri.bundle_path}`;
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function isFourier(x: unknown): x is FourierBundle {
  const o = x as FourierBundle;
  return (
    !!o &&
    typeof o === "object" &&
    Array.isArray(o.freqs) &&
    Array.isArray(o.power_mean) &&
    o.freqs.length === o.power_mean.length &&
    Array.isArray(o.per_dim_dominant) &&
    !!o.meta &&
    typeof o.meta.quantity === "string"
  );
}

/** Fetch the pinned bundle and check it byte-for-byte against the manifest.
 *  Only a verified bundle is handed to the chart; anything else is reported
 *  with what was expected and what was found, and nothing is substituted. */
export async function verifyBundle(
  ri: ResearchIntro,
  fetchImpl: typeof fetch = fetch,
  base = DATA_BASE,
): Promise<Exclude<BundleCheck, { state: "pending" }>> {
  const url = bundleUrl(ri, base);
  let res: Response;
  try {
    res = await fetchImpl(url, { cache: "no-cache" });
  } catch (e) {
    return { state: "missing", url, status: null, message: e instanceof Error ? e.message : String(e) };
  }
  if (!res.ok) return { state: "missing", url, status: res.status, message: `the server answered ${res.status}` };
  const bytes = new Uint8Array(await res.arrayBuffer());
  const actual = await sha256Hex(bytes);
  if (actual !== ri.sha256 || bytes.length !== ri.bytes)
    return { state: "mismatch", url, expected: ri.sha256, actual, bytes: bytes.length };
  let json: unknown;
  try {
    json = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return { state: "mismatch", url, expected: ri.sha256, actual, bytes: bytes.length };
  }
  if (!isFourier(json)) return { state: "mismatch", url, expected: ri.sha256, actual, bytes: bytes.length };
  return { state: "verified", bytes, json, url };
}

/* ── the value table ──────────────────────────────────────────────────── */

export interface FrequencyRow {
  /** index into the bundle's arrays */
  index: number;
  /** cycles per context window */
  freq: number;
  /** positions per cycle: n_ctx / freq */
  period: number;
  /** mean power over the embedding dimensions */
  power: number;
  /** how many embedding dimensions have their strongest pattern here */
  dims: number;
}

/** The strongest `k` non-zero frequencies, by mean power. Frequency 0 is the
 *  mean, which the bundle removed before the transform, so it is never shown. */
export function topFrequencies(b: FourierBundle, k = 8): FrequencyRow[] {
  const counts = new Map<number, number>();
  for (const i of b.per_dim_dominant) counts.set(i, (counts.get(i) ?? 0) + 1);
  const rows: FrequencyRow[] = [];
  for (let i = 0; i < b.freqs.length; i++) {
    const freq = b.freqs[i]!;
    if (!(freq > 0)) continue;
    rows.push({ index: i, freq, period: b.meta.n_ctx / freq, power: b.power_mean[i]!, dims: counts.get(i) ?? 0 });
  }
  rows.sort((a, b2) => b2.power - a.power || a.freq - b2.freq);
  return rows.slice(0, k);
}

export function formatPower(p: number): string {
  if (p === 0) return "0";
  if (p >= 0.01 && p < 1e5) return p.toLocaleString("en-US", { maximumFractionDigits: p >= 100 ? 1 : 3 });
  return p.toExponential(2);
}

export function formatPeriod(p: number): string {
  return Number.isInteger(p) ? String(p) : p.toFixed(1);
}

/* ── the task ─────────────────────────────────────────────────────────── */

export interface Option {
  id: string;
  label: string;
  correct?: boolean;
  why: string;
}

export const QUANTITY_OPTIONS: readonly Option[] = [
  {
    id: "spectrum",
    label: "How strongly GPT-2’s stored position embeddings repeat at each frequency, averaged over the 768 dimensions",
    correct: true,
    why: "Yes. The bundle is the power spectrum of the mean-centred position-embedding matrix (1,024 positions × 768 dimensions), averaged over dimensions, with frequency in cycles per context window.",
  },
  {
    id: "attention",
    label: "How often GPT-2 attends to earlier positions while reading a prompt",
    why: "No. Nothing here was measured on a prompt. The input is the stored weight matrix for positions, so attention is not part of it.",
  },
  {
    id: "activation",
    label: "How strongly a learned feature activates at each position of an input",
    why: "No. There is no input and no learned feature in this bundle. It is a transform of fixed weights.",
  },
];

export const CAVEAT_OPTIONS: readonly Option[] = [
  {
    id: "power",
    label: "How much power the position embeddings have at a given frequency",
    why: "The chart does show this: it is the gold curve and the power column.",
  },
  {
    id: "use",
    label: "Whether GPT-2 uses a given frequency when it processes your prompt",
    correct: true,
    why: "Right. These are static weight patterns. How a pattern is used on a prompt would need measurements on activations, which this bundle does not contain. Averaging over dimensions can also hide differences between them.",
  },
  {
    id: "dims",
    label: "How many dimensions have their strongest pattern at a given frequency",
    why: "The chart does show this: it is the cyan bars and the dimensions column.",
  },
];

export const TASK_STEPS = ["quantity", "value", "caveat", "export"] as const;
export type TaskStep = (typeof TASK_STEPS)[number];

export interface IntroProgress {
  /** progress belongs to one bundle digest; any other resets it */
  sha256: string;
  quantity: string | null;
  /** the recorded frequency (bundle index) */
  value: number | null;
  caveat: string | null;
  exported: boolean;
}

export function emptyIntroProgress(sha256: string): IntroProgress {
  return { sha256, quantity: null, value: null, caveat: null, exported: false };
}

const correctId = (opts: readonly Option[]) => opts.find((o) => o.correct)!.id;

export function stepDone(step: TaskStep, p: IntroProgress, rows: readonly FrequencyRow[]): boolean {
  switch (step) {
    case "quantity":
      return p.quantity === correctId(QUANTITY_OPTIONS);
    case "value":
      return p.value !== null && rows.some((r) => r.index === p.value);
    case "caveat":
      return p.caveat === correctId(CAVEAT_OPTIONS);
    case "export":
      return p.exported;
  }
}

export function pendingSteps(p: IntroProgress, rows: readonly FrequencyRow[]): TaskStep[] {
  return TASK_STEPS.filter((s) => !stepDone(s, p, rows));
}

const KEY = "nebulai.research.intro";

export function loadIntroProgress(sha256: string): IntroProgress {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return emptyIntroProgress(sha256);
    const o = JSON.parse(raw) as Partial<IntroProgress>;
    if (o.sha256 !== sha256) return emptyIntroProgress(sha256);
    const str = (v: unknown) => (typeof v === "string" && /^[a-z]{1,20}$/.test(v) ? v : null);
    return {
      sha256,
      quantity: str(o.quantity),
      value: Number.isInteger(o.value) && (o.value as number) >= 0 ? (o.value as number) : null,
      caveat: str(o.caveat),
      exported: o.exported === true,
    };
  } catch {
    return emptyIntroProgress(sha256);
  }
}

export function saveIntroProgress(p: IntroProgress): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* storage blocked: progress lasts for this page only */
  }
}

/** Save the verified bytes exactly as published, under a name that says which
 *  model and file they are. */
export function downloadBytes(bytes: Uint8Array, name: string): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([bytes as Uint8Array<ArrayBuffer>], { type: "application/json" }));
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
