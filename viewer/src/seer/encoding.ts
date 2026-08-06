/** encoding.ts — the one place that decides what a pixel means.
 *
 *  The live view draws the same run several ways: lanes on a time axis, a row
 *  in a fleet strip, a node in a span tree, a particle in an emissive field.
 *  Those are different geometries over one event stream, so the *geometry* can
 *  differ freely — but the moment two of them disagree about what a colour or a
 *  texture means, the page is lying to whoever glances between them. This
 *  module exists so that cannot happen: every projection imports its hues,
 *  caps and textures from here, and `seer-encoding.test.ts` fails if the
 *  contract gains a member with no encoding.
 *
 *  The channel assignment, once, so it is not re-decided per view:
 *
 *    hue        action      what kind of thing the agent did
 *    texture    fidelity    how much we actually know about it
 *    end cap    effect      what it changed
 *    length     duration    only ever a *closed* span's; see `isProvisional`
 *    thickness  magnitude   rank-normalised, never raw; see `rankNormalise`
 *
 *  Two of these carry the subsystem's honesty rules into the render layer:
 *
 *  · **`missing` is drawn as a hole, not a short bar.** Its texture is
 *    `outline` — an unfilled shape — because a filled shape of any length is a
 *    claim about magnitude, and the whole point of `missing` is that we have no
 *    magnitude to claim. `dropped_by_policy` gets its own `policy` texture and
 *    its own ink for the same reason the contract keeps the two apart: one is a
 *    gap in the agent, the other is a setting on our side, and they call for
 *    different actions from whoever is looking.
 *
 *  · **Nothing here reads `native_type`.** The Python analyses are forbidden
 *    from touching it and the same reasoning applies to encoding: an agent's
 *    own vocabulary driving our colours would make two agents' identical work
 *    render differently. Hues key off `action`, which is ours.
 */

import {
  ABSENT_FIDELITIES,
  type Action,
  type Effect,
  type Fidelity,
  type SessionState,
} from "./contract";

/** Hue per action. The lane order in `ACTIONS` roughly follows a healthy run's
 *  path, and these run cool-to-warm along it so a trajectory that drifts from
 *  reading into thrashing warms up visibly. */
export const ACTION_COLOR: Record<Action, string> = {
  inspect: "#5cc7ed",
  search: "#62d9c0",
  edit: "#7dde96",
  execute: "#c782f0",
  verify: "#f5bf5c",
  vcs: "#b0a6f0",
  delegate: "#f090c8",
  interact: "#f07896",
  report: "#969eb5",
};

/** Hue per session state, for the fleet ribbon and the run-list dot.
 *
 *  `waiting_permission` and `waiting_clarification` are deliberately loud: both
 *  mean the agent has stopped and is waiting on a human who may not know it.
 *  That is the one condition where a glance at a fleet of runs has to find the
 *  stalled one immediately. */
export const STATE_COLOR: Record<SessionState, string> = {
  starting: "#686c76",
  idle: "#4a4e58",
  model_running: "#5cc7ed",
  tool_running: "#7dde96",
  waiting_permission: "#f07896",
  waiting_clarification: "#f5bf5c",
  waiting_user: "#f5bf5c",
  compacting: "#c782f0",
  interrupted: "#f5b13d",
  completed: "#3ecf8e",
  failed: "#ff5c7a",
  detached: "#686c76",
};

/** Ink for anything with no action of its own — an unclassified span, a run
 *  whose state we have not learned yet. Deliberately the same dead grey as
 *  `idle`: "nothing is happening here" is what it means in both places. */
export const NEUTRAL_INK = "#686c76";

/** How a span's trailing end is drawn. The effect is the *result* of the work,
 *  so it belongs at the end of the mark that represents it, where the eye
 *  arrives last. */
export type EffectCap =
  /** the work produced something we did not have — a filled wedge */
  | "wedge"
  /** the work changed the world — a solid block */
  | "block"
  /** the work ran and found nothing new — a thin tick */
  | "tick"
  /** the work ran and changed nothing — a flat end, the visual null */
  | "flat"
  /** the work failed — a cross */
  | "cross"
  /** we do not know how it ended — hollow, matching the `missing` texture */
  | "hollow";

export const EFFECT_CAP: Record<Effect, EffectCap> = {
  new_information: "wedge",
  no_new_information: "tick",
  state_changed: "block",
  no_state_change: "flat",
  failed: "cross",
  unknown: "hollow",
};

/** How a mark is filled, given how much we know about it.
 *
 *  `outline` and `policy` are the two that must never render as a filled bar of
 *  some length — see the module note. */
export type Texture =
  /** the agent said so, or we computed it from what the agent said */
  | "solid"
  /** approximate — do not difference it against a solid one */
  | "hatched"
  /** inferred from a pattern; may be wrong */
  | "dotted"
  /** the agent never reported it: an unfilled shape, never a short one */
  | "outline"
  /** we chose not to capture it: unfilled, and visibly *ours* */
  | "policy";

export const FIDELITY_TEXTURE: Record<Fidelity, Texture> = {
  native: "solid",
  deterministic: "solid",
  estimated: "hatched",
  heuristic: "dotted",
  missing: "outline",
  dropped_by_policy: "policy",
};

/** Ink for the two absent fidelities, which are drawn in their own colours
 *  rather than in the action's hue. An absent value tinted like the work it
 *  failed to measure reads as a very small measurement. */
export const ABSENT_INK: Record<"missing" | "dropped_by_policy", string> = {
  missing: "#4a4e58",
  dropped_by_policy: "#5a5470",
};

/** The hue for a mark. Absent fidelities take their own ink, so a hole in the
 *  data never wears the colour of the work it is missing from. */
export function markInk(action: Action | null | undefined, fidelity: Fidelity): string {
  if (fidelity === "missing" || fidelity === "dropped_by_policy") return ABSENT_INK[fidelity];
  return action ? ACTION_COLOR[action] : NEUTRAL_INK;
}

/** Ink for a session state.
 *
 *  Takes a bare string because `time_in_state` arrives from the server keyed by
 *  strings, and a state we have no colour for has to degrade to neutral rather
 *  than to `undefined` — an undefined CSS variable is simply dropped, and the
 *  element then renders in the inherited colour and looks deliberate. */
export function stateInk(state: string | null | undefined): string {
  if (!state) return NEUTRAL_INK;
  return STATE_COLOR[state as SessionState] ?? NEUTRAL_INK;
}

/** True when a mark's extent is not a measurement and must not be read as one.
 *
 *  Two ways that happens, and they are different facts: the span is still
 *  running (its end has not happened yet), or its duration came back absent
 *  (it ended, but nobody clocked it — every reconciled run is like this,
 *  because thread history carries no per-item clock). Both are drawn with a
 *  live cap rather than a measured one; neither may be drawn as a length the
 *  eye can compare against a closed span's. */
export function isProvisional(m: { endedAt?: number | null; fidelity?: Fidelity }): boolean {
  if (m.endedAt == null) return true;
  return m.fidelity != null && ABSENT_FIDELITIES.has(m.fidelity);
}

/** Rank-normalise to [`RANK_FLOOR`, 1].
 *
 *  Thickness and glow both imply magnitude, and both are read comparatively —
 *  "that one is bigger" — rather than absolutely. Raw magnitudes through
 *  either channel are dominated by outliers: one 200k-token turn flattens
 *  every other turn in the run to an invisible hairline, which reads as "those
 *  did nothing" rather than "those were smaller". Ranking spends the channel on
 *  the ordering, which is the part anyone can actually read off a picture.
 *
 *  The floor exists because 0 means "absent" elsewhere in this subsystem; the
 *  smallest *present* value has to stay visibly present. Ties share the mean of
 *  the ranks they span, so two equal values are never drawn differently.
 *
 *  Returns positions in the input's own order. Non-finite inputs get the floor:
 *  they have no place in an ordering. */
export const RANK_FLOOR = 0.12;

export function rankNormalise(values: readonly number[]): number[] {
  const n = values.length;
  if (n === 0) return [];
  if (n === 1) return [Number.isFinite(values[0]!) ? 1 : RANK_FLOOR];

  const order = values
    .map((v, i) => ({ v, i }))
    .filter((e) => Number.isFinite(e.v))
    .sort((a, b) => a.v - b.v);

  const out = new Array<number>(n).fill(RANK_FLOOR);
  if (order.length === 0) return out;
  if (order.length === 1) {
    out[order[0]!.i] = 1;
    return out;
  }

  const span = 1 - RANK_FLOOR;
  const last = order.length - 1;
  let k = 0;
  while (k < order.length) {
    // Tie block: every entry with this value shares the mean rank, so equal
    // inputs cannot come out as different thicknesses.
    let j = k;
    while (j + 1 < order.length && order[j + 1]!.v === order[k]!.v) j++;
    const mean = (k + j) / 2;
    const pos = RANK_FLOOR + (mean / last) * span;
    for (let t = k; t <= j; t++) out[order[t]!.i] = pos;
    k = j + 1;
  }
  return out;
}
