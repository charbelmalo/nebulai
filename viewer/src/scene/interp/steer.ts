/** The arithmetic and the encoding decisions behind #26, in a module with no
 *  GPU in it.
 *
 *  `SteerDriver` is WebGPU-bound and cannot be exercised in a unit test, so
 *  everything the view actually CLAIMS lives here instead: the one
 *  normalization that both column height and colour read, the floor that keeps
 *  a measured zero visible, the control's separate colour, and the summary the
 *  stat strip prints. A claim nothing can check is a claim nothing is holding
 *  up — and the sharpest of these (the α = 0 row is a control, not an absent
 *  measurement) is exactly the kind that quietly rots if only a screenshot
 *  enforces it.
 *
 *  `chrome/SteerRail.tsx` reads the same module, so the text half and the
 *  drawn half cannot drift about which cell is which or what α = 0 means.
 */

import type { InterveneBundle, InterveneRow, InterveneRun } from "../../data/interp";

/* ── the selection channel ─────────────────────────────────────────────────
 *
 *  One selected cell, three surfaces: the stage draws it, `chrome/SteerRail`
 *  reads it, and an episode step sets it. It lives HERE rather than in the
 *  driver so that the two consumers which have no business importing WebGPU —
 *  the chrome rail and `chrome/tours.ts` — do not drag `three/webgpu` into the
 *  boot bundle just to name a cell.
 *
 *  A module-level channel rather than a store slice because the payload is a
 *  whole sweep bundle; pushing 57 KB of generations through `appStore` would
 *  put it in every subscriber's diff on every hover. */

/** Which cell of the sweep is selected: an alpha row and a prompt column. */
export interface SteerCell {
  /** alpha index */
  row: number;
  /** prompt index */
  col: number;
}

export type SteerListener = (b: InterveneBundle | null, sel: SteerCell | null) => void;

const listeners = new Set<SteerListener>();
let current: InterveneBundle | null = null;
let currentSel: SteerCell | null = null;
/** Set by the mounted driver so a selection reaches the GPU columns too. */
let applySel: ((sel: SteerCell) => void) | null = null;

export function onSteer(fn: SteerListener): () => void {
  listeners.add(fn);
  fn(current, currentSel);
  return () => {
    listeners.delete(fn);
  };
}

export function publishSteer(b: InterveneBundle | null, sel: SteerCell | null): void {
  current = b;
  currentSel = sel;
  for (const fn of listeners) fn(b, sel);
}

/** The driver registers on init and clears on dispose. Returns the unregister
 *  so the driver cannot leak a handler into the next feature's session. */
export function registerSteerApply(fn: (sel: SteerCell) => void): () => void {
  applySel = fn;
  return () => {
    if (applySel === fn) applySel = null;
  };
}

/** Select a cell. With a driver mounted the stage moves too; with none (unit
 *  tests, a WebGPU-less rung, an episode step that runs before the canvas is
 *  up) the selection still publishes, so the text half keeps working — and the
 *  request is remembered, so the driver adopts it when it does arrive rather
 *  than opening on its default and silently discarding the step's intent. */
export function selectSteer(sel: SteerCell): void {
  if (applySel) {
    applySel(sel);
    return;
  }
  pending = sel;
  publishSteer(current, sel);
}

let pending: SteerCell | null = null;

/** Consumed once, by the driver, when its bundle lands. Taking rather than
 *  reading is deliberate: a cell requested for one sweep must not silently
 *  reappear as the opening cell of the next one. */
export function takePendingSteer(): SteerCell | null {
  const p = pending;
  pending = null;
  return p;
}

export function steerRun(
  b: InterveneBundle | null,
  sel: SteerCell | null,
): InterveneRun | null {
  if (!b || !sel) return null;
  return b.rows[sel.row]?.runs[sel.col] ?? null;
}

/** A floor so a genuinely-zero column still has a visible plate. "This cell was
 *  measured and came out 0" and "this cell is not in the grid" must never look
 *  the same, and the control row is the one that would otherwise vanish. */
export const FLOOR_FRAC = 0.006;

/** Dark → blue → gold, the family the other ChartStage views use. The bottom
 *  stop is lifted off the page ground on purpose: a small real effect should
 *  still be a solid you can point at, not a smudge. */
export const RAMP: ReadonlyArray<readonly [number, readonly [number, number, number]]> = [
  [0.0, [32, 40, 68]],
  [0.3, [52, 70, 126]],
  [0.58, [70, 150, 214]],
  [0.82, [232, 160, 60]],
  [1.0, [250, 208, 112]],
];

/** The control's own colour. It is NOT a faint measurement at the cool end of
 *  the ramp — it is the zero the ramp is measured from, and colouring it like a
 *  small effect is exactly the confusion this view exists to prevent. */
export const CONTROL_RGB: readonly [number, number, number] = [118, 126, 158];

export function ramp(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  for (let s = 1; s < RAMP.length; s++) {
    const [p0, c0] = RAMP[s - 1]!;
    const [p1, c1] = RAMP[s]!;
    if (x <= p1) {
      const u = p1 === p0 ? 0 : (x - p0) / (p1 - p0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * u),
        Math.round(c0[1] + (c1[1] - c0[1]) * u),
        Math.round(c0[2] + (c1[2] - c0[2]) * u),
      ];
    }
  }
  const last = RAMP[RAMP.length - 1]![1];
  return [last[0], last[1], last[2]];
}

/** The one normalization, shared by height and colour. LINEAR in bits: KL is
 *  already a log quantity, and logging it again would flatten exactly the
 *  differences the sweep was run to show. Against the bundle's OWN maximum,
 *  which is why the height axis prints that maximum — two sweeps must never be
 *  compared by eye merely because their tallest columns reach the same place. */
export function norm(kl: number, maxKl: number): number {
  if (!(maxKl > 0) || !Number.isFinite(kl)) return 0;
  return Math.max(0, Math.min(1, kl / maxKl));
}

/** Cage height from the grid, clamped so a 2-alpha sweep is not a pillar and a
 *  9-alpha one is not a pancake. */
export const cageHeight = (rows: number): number => Math.min(12, Math.max(4.5, rows * 1.6));

/** World-unit height of one column. The floor is applied AFTER normalization,
 *  so a zero reading gets a plate rather than a phantom value. */
export function barHeight(kl: number, maxKl: number, rows: number): number {
  return Math.max(FLOOR_FRAC, norm(kl, maxKl)) * cageHeight(rows);
}

/** The bundle's own maximum KL, over every cell including the control. */
export function maxKl(b: InterveneBundle): number {
  let m = 0;
  for (const r of b.rows) for (const run of r.runs) m = Math.max(m, run.kl_bits);
  return m;
}

export type ControlStatus = "identical" | "broken" | "absent";

/** What this sweep's control actually says. Three states, never two: a bundle
 *  with no α = 0 row is not the same as one whose α = 0 row failed, and both
 *  differ from the one case in which the other columns mean anything. */
export function controlStatus(b: InterveneBundle): ControlStatus {
  const ctrl = b.rows.find((r) => r.is_identity);
  if (!ctrl) return "absent";
  return ctrl.identical_to_baseline ? "identical" : "broken";
}

/** Largest KL over the cells that actually installed a hook. The control is
 *  excluded because it is the zero, not a competitor for the peak. */
export function peakKl(b: InterveneBundle): number | null {
  const moved = b.rows.filter((r) => !r.is_identity);
  if (moved.length === 0) return null;
  return moved.reduce((m, r) => Math.max(m, r.kl_bits_max), 0);
}

/** α formatted so 0 reads as "0" and not "0.00" — the control's label should be
 *  the shortest thing on the axis. */
export function fmtAlpha(a: number): string {
  if (a === 0) return "0";
  if (Number.isInteger(a)) return String(a);
  return a.toFixed(2).replace(/0$/, "");
}

/** Prompts are sentences and the axis label is ~14 characters wide. Truncate on
 *  a word boundary with an ellipsis, never mid-word: a clipped label that looks
 *  like a whole word names a prompt that does not exist. */
export function shortPrompt(p: string, max = 16): string {
  const t = p.trim();
  if (t.length <= max) return t;
  const cut = t.slice(0, max);
  const sp = cut.lastIndexOf(" ");
  return `${(sp > 4 ? cut.slice(0, sp) : cut).trimEnd()}…`;
}

/** How many leading tokens two generations share. Drawn once rather than twice
 *  by the rail: the eye should land on the first place the runs part. */
export function commonPrefix(a: readonly string[], b: readonly string[]): number {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
}

/** The axis label for one alpha row — the control says so ON the axis, not only
 *  in the legend, because the axis is what a cropped screenshot keeps. */
export function alphaLabel(row: InterveneRow | undefined): string {
  if (!row) return "";
  return row.is_identity ? `α ${fmtAlpha(row.alpha)} · control` : `α ${fmtAlpha(row.alpha)}`;
}
