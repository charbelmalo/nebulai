/**
 * Signal conditioning for the hand rig.
 *
 * A raw MediaPipe landmark stream is not usable as a control signal. It arrives
 * at camera cadence (~30 Hz here, not render cadence), it carries per-frame
 * estimator noise of roughly a millimetre, and it is already 45-90 ms behind
 * the hand by the time the inference completes. Feeding it straight into a
 * camera angle or a uniform produces the characteristic "puppet on strings"
 * feel: laggy, jittery and twitchy at threshold crossings.
 *
 * The four tools below are the standard fixes, one per failure mode:
 *
 *   - `oneEuroStep`  jitter, without paying the latency a fixed low-pass costs
 *   - `oneEuroLead`  the residual transport delay, using the filter's own
 *                    velocity estimate as a predictor
 *   - `springStep`   position-coupling — a hand is a *force* on the atlas, not
 *                    a direct handle on it, so releasing coasts and settles
 *   - `schmittStep`  threshold chatter on discrete engagements (pinch, fist)
 *
 * Everything here is plain state-plus-step rather than classes, so each step
 * function can be exercised on a synthetic sample sequence with no camera
 * present — which is exactly what `tests/unit/hands-filters.test.ts` does.
 */

export function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return value < min ? min : value > max ? max : value;
}

export function lerp(from: number, to: number, alpha: number): number {
  return from + (to - from) * alpha;
}

/**
 * The largest timestep the integrators will honour. A backgrounded tab, a
 * garbage-collection pause or a shader recompile can hand us a delta of several
 * seconds; integrating that in one go throws the springs across their range.
 * Clamping is the difference between "the rig hiccups" and "the rig explodes".
 */
const MAX_STEP_SECONDS = 0.1;

/** Springs are substepped to this ceiling so stiff settings stay stable. */
const SPRING_SUBSTEP_SECONDS = 1 / 120;

// ---------------------------------------------------------------------------
// One Euro filter
// ---------------------------------------------------------------------------

export interface OneEuroConfig {
  /** Cutoff in Hz at zero speed. Lower is smoother and laggier when still. */
  minCutoff: number;
  /** How aggressively the cutoff opens with speed. Higher tracks faster. */
  beta: number;
  /** Cutoff in Hz for the internal derivative estimate. */
  derivativeCutoff: number;
}

export interface OneEuroState extends OneEuroConfig {
  initialised: boolean;
  /** Filtered value. */
  value: number;
  /** Last raw sample, needed for the finite difference. */
  previousSample: number;
  /** Filtered derivative, in units per second. Doubles as the lead predictor. */
  velocity: number;
}

export const DEFAULT_ONE_EURO: OneEuroConfig = {
  minCutoff: 1.15,
  beta: 0.05,
  derivativeCutoff: 1,
};

export function createOneEuro(config: Partial<OneEuroConfig> = {}): OneEuroState {
  return {
    ...DEFAULT_ONE_EURO,
    ...config,
    initialised: false,
    value: 0,
    previousSample: 0,
    velocity: 0,
  };
}

/**
 * Exponential smoothing coefficient for a given cutoff frequency and timestep.
 * This is the standard 1€ formulation: the time constant is derived from the
 * cutoff, then converted to an alpha against the *actual* delta rather than an
 * assumed frame rate — which is what lets one filter serve a camera stream
 * whose cadence wanders between 24 and 60 Hz.
 */
function smoothingAlpha(cutoffHz: number, dt: number): number {
  const timeConstant = 1 / (2 * Math.PI * Math.max(cutoffHz, 1e-4));
  return 1 / (1 + timeConstant / Math.max(dt, 1e-5));
}

export function oneEuroStep(state: OneEuroState, sample: number, dt: number): number {
  if (!Number.isFinite(sample)) return state.value;

  if (!state.initialised) {
    state.initialised = true;
    state.value = sample;
    state.previousSample = sample;
    state.velocity = 0;
    return sample;
  }

  const step = clamp(dt, 1e-4, MAX_STEP_SECONDS);
  const rawDerivative = (sample - state.previousSample) / step;
  state.previousSample = sample;

  const derivativeAlpha = smoothingAlpha(state.derivativeCutoff, step);
  state.velocity = lerp(state.velocity, rawDerivative, derivativeAlpha);

  // The whole point of 1€: the cutoff is not fixed. When the hand is still the
  // cutoff collapses and the noise disappears; when the hand moves the cutoff
  // opens and the filter gets out of the way, so smoothing costs latency only
  // where latency is imperceptible.
  const cutoff = state.minCutoff + state.beta * Math.abs(state.velocity);
  state.value = lerp(state.value, sample, smoothingAlpha(cutoff, step));
  return state.value;
}

/**
 * The filtered value projected forward by its own velocity estimate.
 *
 * MediaPipe's landmarks describe where the hand *was* when the frame was
 * captured, not where it is now. Extrapolating along the filtered derivative
 * cancels most of that, and because the derivative is itself 1€-filtered the
 * prediction stays quiet when the hand is still — which is exactly when a naive
 * predictor would overshoot and buzz. `leadSeconds` is deliberately small; past
 * roughly 60 ms the prediction error costs more than the latency it buys.
 */
export function oneEuroLead(state: OneEuroState, leadSeconds: number): number {
  if (!state.initialised) return state.value;
  return state.value + state.velocity * leadSeconds;
}

// ---------------------------------------------------------------------------
// Vector form
// ---------------------------------------------------------------------------

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface OneEuroVec3State {
  x: OneEuroState;
  y: OneEuroState;
  z: OneEuroState;
}

export function createOneEuroVec3(config: Partial<OneEuroConfig> = {}): OneEuroVec3State {
  return {
    x: createOneEuro(config),
    y: createOneEuro(config),
    z: createOneEuro(config),
  };
}

/**
 * Filters a vector with a cutoff shared across all three components.
 *
 * Filtering each axis independently is wrong for a direction: each axis picks
 * its own cutoff from its own speed, so a diagonal sweep smooths x and y by
 * different amounts and the vector swings off its true path before arriving.
 * Driving all three from the vector's speed keeps the components coherent.
 */
export function oneEuroVec3Step(
  state: OneEuroVec3State,
  sample: Vec3,
  dt: number,
  out: Vec3,
): Vec3 {
  const step = clamp(dt, 1e-4, MAX_STEP_SECONDS);

  if (!state.x.initialised) {
    oneEuroStep(state.x, sample.x, step);
    oneEuroStep(state.y, sample.y, step);
    oneEuroStep(state.z, sample.z, step);
    out.x = state.x.value;
    out.y = state.y.value;
    out.z = state.z.value;
    return out;
  }

  const dx = (sample.x - state.x.previousSample) / step;
  const dy = (sample.y - state.y.previousSample) / step;
  const dz = (sample.z - state.z.previousSample) / step;
  const derivativeAlpha = smoothingAlpha(state.x.derivativeCutoff, step);

  state.x.velocity = lerp(state.x.velocity, dx, derivativeAlpha);
  state.y.velocity = lerp(state.y.velocity, dy, derivativeAlpha);
  state.z.velocity = lerp(state.z.velocity, dz, derivativeAlpha);
  state.x.previousSample = sample.x;
  state.y.previousSample = sample.y;
  state.z.previousSample = sample.z;

  const speed = Math.hypot(state.x.velocity, state.y.velocity, state.z.velocity);
  const alpha = smoothingAlpha(state.x.minCutoff + state.x.beta * speed, step);

  state.x.value = lerp(state.x.value, sample.x, alpha);
  state.y.value = lerp(state.y.value, sample.y, alpha);
  state.z.value = lerp(state.z.value, sample.z, alpha);

  out.x = state.x.value;
  out.y = state.y.value;
  out.z = state.z.value;
  return out;
}

export function oneEuroVec3Lead(state: OneEuroVec3State, leadSeconds: number, out: Vec3): Vec3 {
  out.x = oneEuroLead(state.x, leadSeconds);
  out.y = oneEuroLead(state.y, leadSeconds);
  out.z = oneEuroLead(state.z, leadSeconds);
  return out;
}

// ---------------------------------------------------------------------------
// Second-order spring
// ---------------------------------------------------------------------------

export interface SpringState {
  value: number;
  velocity: number;
}

export interface SpringConfig {
  /** Undamped natural frequency in Hz. */
  frequency: number;
  /** Damping ratio. 1 is critical; below 1 overshoots and settles. */
  damping: number;
}

export function createSpring(value = 0): SpringState {
  return { value, velocity: 0 };
}

/**
 * Semi-implicit Euler integration of a damped harmonic oscillator.
 *
 * This is the single most important difference between a hand demo that feels
 * like magic and one that feels like a puppet. Mapping hand position directly
 * onto a parameter means the parameter inherits every tremor in the operator's
 * arm and stops dead the instant tracking drops. Treating the hand as a target
 * that a mass is pulled toward gives the parameter its own momentum: it leads
 * into motion, it coasts when the hand stops, and it settles rather than
 * snapping. A damping ratio a little under 1 (0.7 by default here) produces one
 * small overshoot, which is what reads as weight.
 */
export function springStep(
  state: SpringState,
  target: number,
  dt: number,
  config: SpringConfig,
): number {
  if (!Number.isFinite(target)) return state.value;

  let remaining = clamp(dt, 0, MAX_STEP_SECONDS);
  const omega = 2 * Math.PI * Math.max(config.frequency, 1e-3);
  const damping = Math.max(config.damping, 0);

  while (remaining > 0) {
    const step = Math.min(remaining, SPRING_SUBSTEP_SECONDS);
    remaining -= step;
    const acceleration =
      omega * omega * (target - state.value) - 2 * damping * omega * state.velocity;
    state.velocity += acceleration * step;
    state.value += state.velocity * step;
  }

  return state.value;
}

/** True when a spring has effectively arrived and can stop being integrated. */
export function springAtRest(
  state: SpringState,
  target: number,
  valueEpsilon: number,
  velocityEpsilon = valueEpsilon * 4,
): boolean {
  return (
    Math.abs(state.value - target) <= valueEpsilon &&
    Math.abs(state.velocity) <= velocityEpsilon
  );
}

// ---------------------------------------------------------------------------
// Schmitt trigger
// ---------------------------------------------------------------------------

export interface SchmittState {
  engaged: boolean;
  /** Seconds the current candidate state has been continuously satisfied. */
  dwell: number;
}

export function createSchmitt(): SchmittState {
  return { engaged: false, dwell: 0 };
}

/**
 * Hysteresis with a dwell requirement, for engagements read off a noisy value.
 *
 * A single threshold on a noisy signal chatters: sit a hair either side of it
 * and the pinch fires dozens of times a second. Two thresholds fix the chatter
 * (engage tighter than you release), and the dwell fixes the remaining problem
 * — that a hand passing *through* a pinch on its way somewhere else should not
 * count as one.
 *
 * `value` is expected to fall as the gesture engages (a pinch gap shrinking),
 * so `engageBelow` must be the tighter of the two thresholds.
 */
export function schmittStep(
  state: SchmittState,
  value: number,
  dt: number,
  engageBelow: number,
  releaseAbove: number,
  dwellSeconds = 0,
): boolean {
  const wants = state.engaged ? value < releaseAbove : value < engageBelow;

  if (wants === state.engaged) {
    state.dwell = 0;
    return state.engaged;
  }

  state.dwell += Math.max(dt, 0);
  // Releasing is deliberately not dwell-gated. Holding a grab that the operator
  // has already let go of is far worse than dropping one a frame early.
  if (state.engaged || state.dwell >= dwellSeconds) {
    state.engaged = wants;
    state.dwell = 0;
  }

  return state.engaged;
}
