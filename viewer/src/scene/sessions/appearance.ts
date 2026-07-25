/** appearance.ts — the user-tunable look of the Sessions particle field.
 *
 *  This is the single source of truth for every knob the Settings →
 *  Appearance → Sessions tab exposes, and the one place the default palette
 *  lives. It is deliberately dependency-light: it imports only the
 *  `ToolCategory` *type* from the parser, so both the store (which owns the
 *  defaults) and the driver (which consumes a live config) can import it
 *  without an import cycle — the driver imports the store, so the store must
 *  not import the driver, and neither may reach the palette through the other.
 *
 *  HONESTY: none of these knobs can make an axis lie. The three axis modes
 *  only choose the asinh bend (auto / off / on); every choice stays monotone
 *  and exactly invertible, so tick labels and tooltips always read true values
 *  and only the spacing changes. See scales.ts.
 */

import { CATEGORY_ORDER, type ToolCategory } from "../../chrome/sessionlog";

/** Per-axis spacing. `auto` lets the data decide (mild spreads stay linear);
 *  `linear` forces a plain axis; `eased` forces the asinh compression on even
 *  when the spread is mild. All three are honest — invertible, monotone. */
export type SessionsAxisMode = "auto" | "linear" | "eased";

export interface SessionsAppearance {
  // ── Field: how a single mote looks ──────────────────────────────────────
  /** Base sprite size multiplier (× the driver's world-unit base). */
  pointSize: number;
  /** Gamma on the output-token RANK. Higher = darker field, sharper spikes. */
  glowContrast: number;
  /** Opacity of the dimmest motes — the floor of the brightness ramp. */
  moteFloor: number;
  /** Emissive multiplier at the top of the ramp (what clears the bloom
   *  threshold). Higher = hotter cores, but additive blending clips to white. */
  glowStrength: number;
  /** How much category hue a quiet turn keeps (saturation floor). */
  saturation: number;

  // ── States: hover / select / sub-agent / failure / dimmed ───────────────
  /** Scale multiplier for the hovered mote. */
  hoverEmphasis: number;
  /** Scale multiplier for the selected (pinned) mote. */
  selectEmphasis: number;
  /** Opacity of a sub-agent (sidechain) turn relative to a main-thread one. */
  subAgentOpacity: number;
  /** Colour failed turns red and floor their glow. Off = a failure looks like
   *  any other turn (its red hue and its brightness boost both drop away). */
  markFailures: boolean;
  /** Minimum glow a failed turn is lifted to when markFailures is on. */
  failureGlow: number;
  /** Opacity of a category dimmed from the legend — never fully hidden. */
  dimmedOpacity: number;

  // ── Trails: the turn-order path ─────────────────────────────────────────
  showTrails: boolean;
  /** Ambient opacity of the whole path (structure, not subject). */
  trailOpacity: number;
  /** Opacity of the lit run around the hovered/pinned turn. */
  trailFocusOpacity: number;
  /** How many turns either side of the active one the focus ramp spans. */
  trailFocusSpan: number;
  /** Ribbon half-width in world units. */
  trailWidth: number;

  // ── Scaffold: frame / grid / probe / labels ─────────────────────────────
  showFrame: boolean;
  frameOpacity: number;
  frameColor: string;
  /** Drop-lines from the hovered/pinned turn to the three walls. */
  showProbe: boolean;
  /** HTML axis tick labels. */
  showLabels: boolean;

  // ── Axes: spacing only, always honest ───────────────────────────────────
  axisTime: SessionsAxisMode;
  axisContext: SessionsAxisMode;
  axisNewContext: SessionsAxisMode;

  // ── Colours ─────────────────────────────────────────────────────────────
  /** Category hue at full strength, as hex — the single source the legend, the
   *  inspector and the field all read, so a chip and a node can never differ. */
  categoryColors: Record<ToolCategory, string>;
  /** What a low-magnitude turn desaturates toward. */
  neutralColor: string;
  /** The one place red is allowed. */
  errorColor: string;
}

/** The default palette, hex. These are the exact values the field shipped with
 *  (the old CATEGORY_RGB / NEUTRAL / ERROR_RGB, quantised to 8-bit). */
export const DEFAULT_CATEGORY_COLORS: Record<ToolCategory, string> = {
  orient: "#5cc7ed",
  plan: "#f5bf5c",
  edit: "#7dde96",
  exec: "#c782f0",
  deliver: "#f07896",
  reflect: "#969eb5",
};
export const DEFAULT_NEUTRAL = "#6b85b8";
export const DEFAULT_ERROR = "#ff544d";
export const DEFAULT_FRAME = "#2c3446";

export const DEFAULT_SESSIONS_APPEARANCE: SessionsAppearance = {
  pointSize: 1,
  glowContrast: 1.7,
  moteFloor: 0.17,
  glowStrength: 2.1,
  saturation: 0.6,

  hoverEmphasis: 1.9,
  selectEmphasis: 2.6,
  subAgentOpacity: 0.5,
  markFailures: true,
  failureGlow: 0.85,
  dimmedOpacity: 0.16,

  showTrails: true,
  trailOpacity: 0.028,
  trailFocusOpacity: 0.5,
  trailFocusSpan: 14,
  trailWidth: 0.0016,

  showFrame: true,
  frameOpacity: 0.5,
  frameColor: DEFAULT_FRAME,
  showProbe: true,
  showLabels: true,

  axisTime: "auto",
  axisContext: "auto",
  axisNewContext: "auto",

  categoryColors: { ...DEFAULT_CATEGORY_COLORS },
  neutralColor: DEFAULT_NEUTRAL,
  errorColor: DEFAULT_ERROR,
};

// ── colour helpers ───────────────────────────────────────────────────────────

/** Parse `#rgb` / `#rrggbb` (with or without the hash) to 0–1 floats. Falls
 *  back to mid-grey on anything unparseable, so a bad stored value can never
 *  blank a layer. */
export function hexToRgb(hex: string): [number, number, number] {
  let h = hex.trim().replace(/^#/, "");
  if (h.length === 3) h = h[0]! + h[0]! + h[1]! + h[1]! + h[2]! + h[2]!;
  if (h.length !== 6 || /[^0-9a-fA-F]/.test(h)) return [0.5, 0.5, 0.5];
  const n = parseInt(h, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export function rgbToHex(rgb: [number, number, number]): string {
  const c = (v: number) =>
    Math.max(0, Math.min(255, Math.round(v * 255)))
      .toString(16)
      .padStart(2, "0");
  return `#${c(rgb[0])}${c(rgb[1])}${c(rgb[2])}`;
}

/** The effective CSS colour for a category under a given config — falls back to
 *  the neutral so an unknown/stored-legacy category still gets a colour. */
export function categoryColor(cfg: SessionsAppearance, cat: ToolCategory): string {
  return cfg.categoryColors[cat] ?? cfg.neutralColor;
}

/** Category colours in CATEGORY_ORDER — the order the driver's per-category
 *  uniforms are indexed in, so a turn's category index picks the right hue. */
export function orderedCategoryRgb(cfg: SessionsAppearance): [number, number, number][] {
  return CATEGORY_ORDER.map((c) => hexToRgb(cfg.categoryColors[c] ?? cfg.neutralColor));
}
