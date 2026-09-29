import { clamp } from "./filters";
import type { HandFeatures, Handedness, SpellEvent, SpellId } from "./types";

/**
 * Discrete gesture recognition: the two visual casts.
 *
 * `channels.ts` answers "where is the camera going right now"; this module
 * answers "did something just *happen*". They are separated because the two
 * kinds of gesture have opposite failure modes and therefore opposite tuning.
 *
 * A continuous channel is forgiving. It is steered, the operator watches the
 * result and corrects mid-motion, and a little noise reads as liveliness. So it
 * is smoothed hard, has no threshold to chatter across, and needs no
 * confirmation that it engaged.
 *
 * A spell is unforgiving in both directions. Firing one the operator did not
 * intend is worse than missing one they did, and a spell that fires with no
 * visible announcement is indistinguishable from a bug.
 *
 * **The hand this module sees is never the hand that is steering.** That is the
 * entire correction, and it is enforced by the caller rather than by a threshold
 * here. The previous design read casts off whichever hands were in frame, which
 * meant the same open palm that was flying the camera was also charging a
 * shockwave, and the same forward push that zoomed in was also the thrust that
 * released it. No refractory period can separate two readings of one motion —
 * only giving them different hands can. With one hand raised, which is the
 * common case, this module is handed `null` and nothing can fire at all.
 *
 * What survives from the old tuning, because it was right:
 *
 *  - **Thresholds sit far above hand tremor, not just above noise.** A resting
 *    arm drifts at roughly 0.05 viewport widths per second and a deliberate
 *    push runs an order of magnitude past that, so the thrust gate is set in
 *    the gap rather than at the noise floor.
 *  - **Every spell has a refractory period.** Without one a single physical
 *    gesture spans several frames above threshold and fires that many times.
 *  - **Charge is spent, not merely read.** A thrust consumes the accumulated
 *    charge so the next one starts from nothing, which is what makes holding
 *    still beforehand feel like it bought something.
 */

/** Seconds of a still open palm to reach full charge. */
const CHARGE_SECONDS = 1.15;

/**
 * Palm speed below which a hand counts as held still, in viewport widths per
 * second. Set above the ~0.05 drift of an unsupported arm so charging does not
 * demand a steadiness nobody has, and well below a deliberate sweep.
 */
const STILL_SPEED = 0.16;

/** How fast charge bleeds away while the palm is moving, in units per second. */
const CHARGE_BLEED = 1.6;

/** How fast charge bleeds away with no free palm present at all. */
const CHARGE_DECAY = 0.8;

/**
 * Approach rate that counts as a thrust, in apparent-size units per second.
 *
 * `screenSpan` runs about 0.08 at arm's length and 0.30 up close, so a push
 * covering that span in a third of a second lands near 0.66. Half of that is
 * comfortably past any reach-and-settle and short of what the operator has to
 * lunge for.
 */
const THRUST_RATE = 0.33;

/** Minimum charge before a thrust is allowed to fire at all. */
const THRUST_MIN_CHARGE = 0.12;

/**
 * Pinch-gap opening rate that counts as a snap, per second.
 *
 * The gap is normalised by hand span and runs 0..1, so a release covering most
 * of it inside 150 ms is about 4. A relaxed opening is under 1.
 */
const SNAP_RATE = 2.6;

/** Per-spell refractory period, in seconds. */
const REFRACTORY: Record<SpellId, number> = {
  // Long enough that one physical push cannot double-fire, short enough to
  // allow a deliberate second wave.
  shockwave: 0.55,
  snap: 0.35,
};

export interface SpellState {
  /** Accumulated free-palm charge, 0..1. */
  charge: number;
  /** Which hand is currently charging, for the overlay. */
  chargingHand: Handedness | null;
  /** Seconds remaining before each spell may fire again. */
  cooldown: Record<SpellId, number>;
}

export function createSpellState(): SpellState {
  return {
    charge: 0,
    chargingHand: null,
    cooldown: { shockwave: 0, snap: 0 },
  };
}

/** Clears the rig's own spell state. */
export function resetSpellState(state: SpellState): void {
  const fresh = createSpellState();
  state.charge = fresh.charge;
  state.chargingHand = null;
  state.cooldown = fresh.cooldown;
}

function ready(state: SpellState, id: SpellId): boolean {
  return state.cooldown[id] <= 0;
}

function fire(
  state: SpellState,
  id: SpellId,
  power: number,
  hand: HandFeatures,
  now: number,
): SpellEvent {
  state.cooldown[id] = REFRACTORY[id];
  return {
    id,
    power: clamp(power, 0, 1),
    origin: { x: hand.palmCentre.x, y: hand.palmCentre.y },
    hand: hand.hand,
    at: now,
  };
}

/**
 * Advances every timer and returns whatever fired this frame.
 *
 * `hand` is the **free** hand — the one not steering the camera — or `null` when
 * there is no such hand, which is both the one-hand case and the effects-off
 * case. Passing a single hand rather than the frame's whole list is deliberate:
 * it makes it impossible for this module to fire off the steering hand even by
 * accident, because it is never shown it.
 *
 * Returns a plain array rather than invoking callbacks so the caller controls
 * ordering and can apply a whole frame's worth of events atomically.
 */
export function updateSpells(
  state: SpellState,
  hand: HandFeatures | null,
  dt: number,
  now: number,
): SpellEvent[] {
  const step = Math.max(dt, 0);
  for (const id of Object.keys(state.cooldown) as SpellId[]) {
    state.cooldown[id] = Math.max(0, state.cooldown[id] - step);
  }

  const events: SpellEvent[] = [];

  // --- Charge ---------------------------------------------------------------
  const palm = hand && (hand.role === "light" || hand.role === "charge") ? hand : null;

  if (palm) {
    state.chargingHand = palm.hand;
    if (palm.speed <= STILL_SPEED) {
      state.charge = clamp(state.charge + step / CHARGE_SECONDS, 0, 1);
    } else {
      // Bleeding rather than zeroing keeps a small correction from throwing
      // away a charge the operator spent a second building.
      state.charge = clamp(state.charge - step * CHARGE_BLEED, 0, 1);
    }
  } else {
    state.chargingHand = null;
    state.charge = clamp(state.charge - step * CHARGE_DECAY, 0, 1);
  }

  if (!hand) return events;

  // --- Impulses -------------------------------------------------------------
  if (
    palm &&
    hand.approachRate >= THRUST_RATE &&
    state.charge >= THRUST_MIN_CHARGE &&
    ready(state, "shockwave")
  ) {
    // Power is mostly charge, with a floor so an uncharged shove still does
    // something, and a small contribution from how hard the push was.
    const vigour = clamp((hand.approachRate - THRUST_RATE) / THRUST_RATE, 0, 1);
    events.push(fire(state, "shockwave", 0.3 + 0.55 * state.charge + 0.15 * vigour, hand, now));
    state.charge = 0;
    return events;
  }

  if (hand.pinchRate >= SNAP_RATE && ready(state, "snap")) {
    const vigour = clamp((hand.pinchRate - SNAP_RATE) / (SNAP_RATE * 1.5), 0, 1);
    events.push(fire(state, "snap", 0.45 + 0.55 * vigour, hand, now));
  }

  return events;
}
