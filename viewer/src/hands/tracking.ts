import type { GestureRecognizer } from "@mediapipe/tasks-vision";

/**
 * The runtime descriptor `createFromOptions` takes.
 *
 * Recovered from the signature rather than imported: the package declares
 * `WasmFileset` but does not export it, on the assumption that every caller
 * obtains one from `FilesetResolver` and never needs to name the type. This rig
 * builds its own — see `resolveWasmFileset` — so it does.
 */
type WasmFileset = Parameters<
  typeof import("@mediapipe/tasks-vision").GestureRecognizer.createFromOptions
>[0];

// The WASM runtime is resolved through the package's own export map and emitted
// as a same-origin, content-hashed asset. It is deliberately *not* loaded from
// a CDN: a deployed viewer may sit behind a host-level Content-Security-Policy
// it cannot see from the checkout, and a same-origin asset is the only version
// that survives one. Note that WebAssembly instantiation additionally requires
// `'wasm-unsafe-eval'` in `script-src` wherever a policy exists at all.
import wasmLoaderUrl from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";
import wasmBinaryUrl from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";
import wasmLoaderNoSimdUrl from "@mediapipe/tasks-vision/vision_wasm_nosimd_internal.js?url";
import wasmBinaryNoSimdUrl from "@mediapipe/tasks-vision/vision_wasm_nosimd_internal.wasm?url";
import {
  createHandChannels,
  handChannelsSettled,
  resetHandChannels,
  setHandEffectsEnabled,
  stepHandChannels,
  updateHandChannelTargets,
  type HandChannelOutput,
  type HandChannelState,
} from "./channels";
import {
  ageHandFilterBank,
  asCannedGesture,
  correctHandedness,
  createHandFilterBank,
  extractHandFeatures,
  type HandFilterBank,
} from "./features";
import type { Lasso, StrokePoint } from "./strokes";
import {
  EMPTY_HAND_DIAGNOSTICS,
  type Handedness,
  type HandFeatures,
  type HandReading,
  type HandRole,
  type HandTrackingDiagnostics,
  type HandTrackingPhase,
  type SpellEvent,
  type SpellId,
} from "./types";

/**
 * Camera-driven hand tracking.
 *
 * The tracker owns four things: the camera stream, the MediaPipe recognizer, the
 * conversion of each inference result into channel *targets*, and the animation
 * frame that integrates the springs toward those targets.
 *
 * The last of those looks misplaced — the springs feed the atlas, so the app's
 * own frame loop in `main.ts` is the obvious home for them — and it is not. That
 * loop only runs while an atlas driver exists; the rig has to keep integrating
 * on the Internals page, on a Seer build, and on a checkout with no baked
 * artifacts at all, or a released grab freezes halfway home and the operator
 * sees the rig break. Springs are integrated here, at display cadence, and the
 * driver only copies the settled values into its camera and uniforms. It is also
 * why they are not integrated on the camera's own callback: at 25-30 Hz, and at
 * that cadence's jitter, a spring puts back most of the stepping the filters
 * were added to remove.
 *
 * Everything is loaded lazily behind the operator's permission grant. The
 * MediaPipe bundle is 152 KB, its WASM runtime 11 MB and the gesture model a
 * further 8 MB; none of that should be paid for by a visitor who never raises a
 * hand, so the import is dynamic and the assets are only referenced from inside
 * it.
 *
 * This is a plain class rather than a hook. The rig has to be reachable from the
 * Preact chrome, from `main.ts`, and from a driver that must never import
 * chrome, and a hook can serve only the first of those.
 */

const MODEL_URL = new URL(
  "models/gesture_recognizer.task",
  new URL(import.meta.env.BASE_URL || "/", location.href),
).href;

/**
 * A modest capture size, on purpose. The landmark model works from a 224 px
 * crop, so a 1280x720 stream costs several times the texture upload per frame
 * for no additional accuracy — and this page is already competing with a WebGPU
 * renderer for the same device.
 */
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const CAPTURE_FPS = 30;

/**
 * Detection is deliberately stricter than tracking.
 *
 * A false *acquisition* costs a phantom hand that grabs the camera; a false
 * *loss* costs one frame of coasting, which the springs absorb invisibly. So the
 * gate to enter is high and the gate to stay is low.
 */
const MIN_DETECTION_CONFIDENCE = 0.6;
const MIN_PRESENCE_CONFIDENCE = 0.5;
const MIN_TRACKING_CONFIDENCE = 0.45;

/** Compiled with `wasm-feature-detect`'s SIMD probe module. */
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253,
  15, 253, 98, 11,
]);

function supportsWasmSimd(): boolean {
  try {
    return WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
}

/**
 * Builds the fileset by hand instead of calling `FilesetResolver.forVisionTasks`.
 *
 * That helper takes a *directory* and appends known filenames, which requires
 * the runtime to sit at a stable path the bundler knows nothing about — in
 * practice, a CDN. Constructing the fileset from hashed asset URLs keeps the
 * whole thing same-origin and cache-busted, at the cost of replicating the one
 * decision the helper makes: which of the two builds this browser can run.
 *
 * The runtime and the `.task` bundle are version-coupled. Both come from
 * `@mediapipe/tasks-vision` at the version pinned in package.json; do not update
 * one without the other.
 */
function resolveWasmFileset(): WasmFileset {
  return supportsWasmSimd()
    ? { wasmLoaderPath: wasmLoaderUrl, wasmBinaryPath: wasmBinaryUrl }
    : { wasmLoaderPath: wasmLoaderNoSimdUrl, wasmBinaryPath: wasmBinaryNoSimdUrl };
}

export interface HandOverlayHand {
  hand: Handedness;
  role: HandRole;
  /** Landmarks in mirrored viewport coordinates, both axes 0..1. */
  points: { x: number; y: number }[];
}

export interface HandOverlayCast {
  id: SpellId;
  x: number;
  y: number;
  age: number;
  power: number;
  /**
   * Monotonic across the session. The announcer needs to know when a cast is
   * *new*, and identity alone cannot tell it — the same spell cast twice in a
   * row is two announcements, not one.
   */
  seq: number;
}

export interface HandOverlay {
  hands: HandOverlayHand[];
  /** The in-progress air stroke, in the same coordinate space. */
  stroke: { x: number; y: number }[];
  /**
   * Casts still worth drawing, newest last.
   *
   * Kept on the overlay rather than handed to the consumer as a one-shot event
   * because the ring an operator sees is an *animation*, and a callback that
   * fires once cannot drive one. The overlay is redrawn every frame anyway, so a
   * short list of recent casts with their ages is all the drawing code needs.
   */
  casts: HandOverlayCast[];
  /** Open-palm charge, 0..1, and where to draw it. */
  charge: number;
  chargeAt: { x: number; y: number } | null;
}

/** How long a cast stays on the overlay after it fires, in seconds. */
const CAST_OVERLAY_SECONDS = 0.9;

export interface HandTrackerStatus {
  phase: HandTrackingPhase;
  error: string | null;
  diagnostics: HandTrackingDiagnostics;
}

export interface HandTrackerOptions {
  /**
   * Whether the free hand may cast visual effects, at construction.
   *
   * Off unless asked for. The rig's job is navigation; effects are a thing the
   * operator opts into once they are comfortable, and defaulting them on is how
   * the previous design ended up firing a shockwave at someone who was trying to
   * pan.
   */
  effects?: boolean;
  /** Fired when an air-drawn lasso resolves. */
  onLasso?: (lasso: Lasso) => void;
  /**
   * Fired when a pointing finger dwells somewhere.
   *
   * The point is in mirrored viewport coordinates; what is *at* that point is
   * the host's problem, because answering it needs the id buffer.
   */
  onWaypoint?: (point: StrokePoint) => void;
  /**
   * Fired once per discrete cast.
   *
   * The channel layer has already applied everything a cast does by the time
   * this runs — both remaining casts are purely visual and live entirely inside
   * that layer — so what reaches the host is only the announcement. It is
   * deliberately no longer a hook for changing the camera or the store: a cast
   * that did either would be navigation wearing a gesture, which is exactly what
   * made the rig unusable.
   */
  onSpell?: (event: SpellEvent) => void;
  /**
   * Receives the sprung channels on every integration step.
   *
   * Already smoothed, so the consumer must apply them immediately rather than
   * feeding them through a second easing stage — two cascaded smoothers read as
   * lag, not as polish.
   */
  onStep?: (output: HandChannelOutput) => void;
  /** Fired whenever phase, error or diagnostics change. */
  onStatus?: (status: HandTrackerStatus) => void;
}

function emptyReading(): HandReading {
  return { timestamp: 0, hands: [], dominant: null, secondary: null };
}

function emptyOverlay(): HandOverlay {
  return { hands: [], stroke: [], casts: [], charge: 0, chargeAt: null };
}

/** The shape `recognizeForVideo` returns, narrowed to what the rig reads. */
interface RecognizerResult {
  landmarks: { x: number; y: number; z: number }[][];
  worldLandmarks: { x: number; y: number; z: number }[][];
  handedness: { categoryName: string; score: number }[][];
  gestures: { categoryName: string; score: number }[][];
}

export class HandTracker {
  /**
   * The preview element.
   *
   * Created here rather than handed in by whichever component happens to mount
   * first. The rig outlives the HUD — it keeps integrating while the panel is
   * closed — and a tracker that fails with "the preview was not mounted"
   * (which the rig this is ported from could) makes the camera's lifetime
   * depend on a render order nobody controls. The HUD adopts this element.
   */
  readonly video: HTMLVideoElement;
  readonly channels: HandChannelState;
  overlay: HandOverlay = emptyOverlay();

  private phase: HandTrackingPhase = "idle";
  private error: string | null = null;
  private diagnostics: HandTrackingDiagnostics = EMPTY_HAND_DIAGNOSTICS;

  private readonly options: HandTrackerOptions;
  private stream: MediaStream | null = null;
  private recognizer: GestureRecognizer | null = null;
  private banks: Record<Handedness, HandFilterBank> = {
    Left: createHandFilterBank(),
    Right: createHandFilterBank(),
  };
  private reading: HandReading = emptyReading();
  private casts: HandOverlayCast[] = [];
  private castSeq = 0;
  private subscribers = new Set<() => void>();

  private springFrame: number | null = null;
  private springLastAt = 0;
  private springClock = 0;

  private running = false;
  private frameHandle: number | null = null;
  private usesVideoCallback = false;
  private lastSampleAt = 0;
  private lastTimestamp = 0;
  private accumulator = { frames: 0, seconds: 0, inferenceMs: 0 };
  private delegate: "GPU" | "CPU" = "GPU";
  /**
   * Bumped by every `enable`/`disable`/`destroy`.
   *
   * `enable` awaits three times — permission, the dynamic import, `video.play()`
   * — and each await is a place the operator can have turned the rig off again.
   * Comparing the generation after every one is what stops a cancelled start
   * from leaving a live camera nobody has a handle on.
   */
  private generation = 0;
  private paused = false;
  private hidden = false;
  private onVisibility: (() => void) | null = null;

  constructor(options: HandTrackerOptions) {
    this.options = options;
    this.channels = createHandChannels(options.effects ?? false);
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.autoplay = true;
    // The HUD renders its own mirrored preview from this element; keeping it out
    // of the accessibility tree here means the raw feed is never announced twice.
    this.video.setAttribute("aria-hidden", "true");

    if (typeof document !== "undefined") {
      this.onVisibility = () => {
        // Inference against a hidden tab is pure heat: the camera keeps running
        // but nothing can be seen to respond to it.
        this.hidden = document.hidden;
      };
      document.addEventListener("visibilitychange", this.onVisibility);
      this.hidden = document.hidden;
    }
  }

  /** Whether this browser can run the rig at all. */
  get available(): boolean {
    return (
      typeof navigator !== "undefined" &&
      typeof navigator.mediaDevices?.getUserMedia === "function" &&
      typeof WebAssembly !== "undefined"
    );
  }

  get status(): HandTrackerStatus {
    return { phase: this.phase, error: this.error, diagnostics: this.diagnostics };
  }

  get enabled(): boolean {
    return this.phase === "tracking";
  }

  /** Announces a new reading or a new integration step. */
  subscribe(listener: () => void): () => void {
    this.subscribers.add(listener);
    return () => {
      this.subscribers.delete(listener);
    };
  }

  /**
   * Turns the free hand's effects vocabulary on or off mid-session.
   *
   * Springs are scheduled afterwards because switching *off* drops a charge and
   * any wave in flight, and the point gain those left behind has to be walked
   * back to neutral — with no hand in frame nothing else would ever run the
   * integrator that does it.
   */
  setEffectsEnabled(enabled: boolean): void {
    setHandEffectsEnabled(this.channels, enabled);
    this.scheduleSprings();
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
  }

  reset(): void {
    resetHandChannels(this.channels);
    this.announce();
    this.scheduleSprings();
  }

  private announce(): void {
    for (const listener of this.subscribers) listener();
  }

  private setStatus(
    phase: HandTrackingPhase,
    error: string | null = this.error,
    diagnostics: HandTrackingDiagnostics = this.diagnostics,
  ): void {
    this.phase = phase;
    this.error = error;
    this.diagnostics = diagnostics;
    this.options.onStatus?.(this.status);
  }

  // ── spring integrator ────────────────────────────────────────────────────

  /**
   * Starts the spring integrator if it is not already running.
   *
   * Called whenever something moves a target — a new inference result, a
   * release, a reset — rather than run unconditionally, so a page with the rig
   * constructed but idle schedules no frames at all.
   */
  private scheduleSprings(): void {
    if (this.springFrame !== null || typeof window === "undefined") return;
    this.springFrame = window.requestAnimationFrame((time) => {
      this.stepSprings(time);
    });
  }

  private stepSprings(time: number): void {
    this.springFrame = null;

    // The first frame after a start has no previous timestamp to difference
    // against, and `time - 0` is the age of the document. Assume a frame.
    const delta =
      this.springLastAt === 0
        ? 1 / 60
        : Math.min(0.1, Math.max(1 / 240, (time - this.springLastAt) / 1000));
    this.springLastAt = time;
    this.springClock += 1;

    const output = stepHandChannels(this.channels, delta, this.springClock);
    this.options.onStep?.(output);
    this.announce();

    if (this.channels.tracking || !handChannelsSettled(this.channels)) {
      this.scheduleSprings();
      return;
    }

    // Settled and nothing in frame: stop scheduling, and forget the timestamp so
    // the next start measures its first delta from a frame rather than from the
    // length of the pause.
    this.springLastAt = 0;
  }

  private stopSprings(): void {
    const handle = this.springFrame;
    if (handle === null || typeof window === "undefined") return;
    this.springFrame = null;
    this.springLastAt = 0;
    window.cancelAnimationFrame(handle);
  }

  // ── inference ────────────────────────────────────────────────────────────

  /**
   * Turns one inference result into channel targets.
   *
   * `dt` is measured between consumed frames rather than assumed, because the
   * camera does not deliver at a fixed rate: it drops frames under load and
   * under poor lighting the exposure time itself lengthens. Every filter and
   * every threshold in the rig is written against real elapsed time for exactly
   * that reason.
   */
  private consume(result: RecognizerResult, now: number): void {
    const dt =
      this.lastSampleAt === 0
        ? 1 / CAPTURE_FPS
        : Math.min(0.25, Math.max(1 / 240, (now - this.lastSampleAt) / 1000));
    this.lastSampleAt = now;

    const hands: HandFeatures[] = [];
    const overlayHands: HandOverlayHand[] = [];
    const seen = new Set<Handedness>();

    for (let index = 0; index < result.landmarks.length; index += 1) {
      const landmarks = result.landmarks[index];
      const world = result.worldLandmarks[index];
      const handedness = result.handedness[index]?.[0];
      if (!landmarks || !world || !handedness) continue;

      const hand = correctHandedness(handedness.categoryName);
      // Two hands can momentarily be labelled the same during a crossover.
      // Whichever arrives first keeps the filter bank; the duplicate is dropped
      // rather than allowed to fight over it.
      if (seen.has(hand)) continue;
      seen.add(hand);

      const gestureCategory = result.gestures[index]?.[0];
      const features = extractHandFeatures({
        landmarks,
        world,
        hand,
        gesture: asCannedGesture(gestureCategory?.categoryName),
        gestureScore: gestureCategory?.score ?? 0,
        bank: this.banks[hand],
        dt,
      });
      if (!features) continue;

      hands.push(features);
      overlayHands.push({
        hand,
        role: features.role,
        points: landmarks.map((point) => ({ x: 1 - point.x, y: point.y })),
      });
    }

    for (const hand of ["Left", "Right"] as const) {
      if (!seen.has(hand)) ageHandFilterBank(this.banks[hand], dt);
    }

    this.reading = {
      timestamp: now,
      hands,
      dominant: hands[0] ?? null,
      secondary: hands[1] ?? null,
    };

    const events = updateHandChannelTargets(this.channels, this.reading, dt);

    // Roles are assigned inside the channel update, so the overlay is filled in
    // afterwards rather than carrying whatever the previous frame had.
    for (const entry of overlayHands) {
      const match = hands.find((candidate) => candidate.hand === entry.hand);
      entry.role = match?.role ?? "idle";
    }

    // Age the casts already on screen, drop the expired ones, then add whatever
    // fired this frame at age zero.
    const casts = this.casts
      .map((cast) => ({ ...cast, age: cast.age + dt }))
      .filter((cast) => cast.age < CAST_OVERLAY_SECONDS);
    for (const spell of events.spells) {
      this.castSeq += 1;
      casts.push({
        id: spell.id,
        x: spell.origin.x,
        y: spell.origin.y,
        age: 0,
        power: spell.power,
        seq: this.castSeq,
      });
    }
    this.casts = casts;

    const chargingHand = this.channels.output.chargingHand;
    const chargeSource = chargingHand
      ? hands.find((candidate) => candidate.hand === chargingHand)
      : undefined;

    this.overlay = {
      hands: overlayHands,
      stroke: (this.channels.output.stroke ?? []).map((point) => ({ x: point.x, y: point.y })),
      casts,
      charge: this.channels.output.charge,
      chargeAt: chargeSource
        ? { x: chargeSource.palmCentre.x, y: chargeSource.palmCentre.y }
        : null,
    };

    if (events.lasso) this.options.onLasso?.(events.lasso);
    if (events.waypoint) this.options.onWaypoint?.(events.waypoint);
    for (const spell of events.spells) this.options.onSpell?.(spell);

    this.announce();
    this.scheduleSprings();
  }

  private stopLoop(): void {
    this.running = false;
    const handle = this.frameHandle;
    if (handle === null) return;
    this.frameHandle = null;

    if (this.usesVideoCallback && this.video.cancelVideoFrameCallback) {
      this.video.cancelVideoFrameCallback(handle);
      return;
    }
    if (typeof window !== "undefined") window.cancelAnimationFrame(handle);
  }

  // ── lifecycle ────────────────────────────────────────────────────────────

  async enable(): Promise<void> {
    if (!this.available) {
      this.setStatus("error", "This browser does not expose a camera to the page.");
      return;
    }
    if (this.recognizer) return;

    const generation = this.generation + 1;
    this.generation = generation;
    this.setStatus("requesting", null);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: CAPTURE_WIDTH },
          height: { ideal: CAPTURE_HEIGHT },
          frameRate: { ideal: CAPTURE_FPS },
        },
      });
    } catch (cause) {
      if (this.generation !== generation) return;
      const name = cause instanceof DOMException ? cause.name : "";
      this.setStatus(
        "error",
        name === "NotAllowedError"
          ? "Camera access was declined. Hand control needs it; nothing leaves this machine."
          : name === "NotFoundError"
            ? "No camera was found on this machine."
            : "The camera could not be opened.",
      );
      return;
    }

    if (this.generation !== generation) {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    this.stream = stream;
    this.setStatus("loading");

    let recognizer: GestureRecognizer;
    this.delegate = "GPU";
    try {
      const vision = await import("@mediapipe/tasks-vision");
      const fileset = resolveWasmFileset();

      const create = (chosen: "GPU" | "CPU") =>
        vision.GestureRecognizer.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: chosen },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: MIN_DETECTION_CONFIDENCE,
          minHandPresenceConfidence: MIN_PRESENCE_CONFIDENCE,
          minTrackingConfidence: MIN_TRACKING_CONFIDENCE,
        });

      try {
        recognizer = await create("GPU");
      } catch {
        // A GPU delegate failure is routine rather than exceptional: it is what
        // a software renderer, a blocklisted driver or an exhausted WebGL
        // context budget looks like from here — and this page has already taken
        // a WebGPU context for the atlas. CPU inference is roughly three times
        // slower and still usable, so it is worth taking silently and reporting
        // in the diagnostics rather than failing the whole rig.
        this.delegate = "CPU";
        recognizer = await create("CPU");
      }
    } catch (cause) {
      if (this.generation !== generation) return;
      for (const track of stream.getTracks()) track.stop();
      this.stream = null;
      this.setStatus(
        "error",
        cause instanceof Error && /wasm|WebAssembly/i.test(cause.message)
          ? "The hand-tracking runtime was blocked. The page CSP must allow wasm-unsafe-eval."
          : "The hand-tracking model failed to load.",
      );
      return;
    }

    if (this.generation !== generation) {
      recognizer.close();
      for (const track of stream.getTracks()) track.stop();
      this.stream = null;
      return;
    }

    this.recognizer = recognizer;
    this.video.srcObject = stream;
    try {
      await this.video.play();
    } catch {
      // Autoplay of a muted, user-initiated stream is permitted everywhere this
      // application runs; if it is refused the frame loop simply sees a video
      // with no dimensions and idles, which the diagnostics make visible.
    }

    if (this.generation !== generation) return;

    this.setStatus("tracking", null);
    // Ahead of the first inference, so the channels are already being integrated
    // when the first reading lands rather than jumping to meet it.
    this.scheduleSprings();
    this.running = true;
    this.usesVideoCallback = typeof this.video.requestVideoFrameCallback === "function";
    this.lastSampleAt = 0;
    this.accumulator = { frames: 0, seconds: 0, inferenceMs: 0 };

    this.tick(generation);
  }

  private tick(generation: number): void {
    if (!this.running || this.generation !== generation) return;

    const recognizer = this.recognizer;
    const video = this.video;
    const schedule = () => {
      if (!this.running || this.generation !== generation) return;
      if (this.usesVideoCallback && video.requestVideoFrameCallback) {
        this.frameHandle = video.requestVideoFrameCallback(() => {
          this.tick(generation);
        });
      } else if (typeof window !== "undefined") {
        this.frameHandle = window.requestAnimationFrame(() => {
          this.tick(generation);
        });
      }
    };

    if (!recognizer || this.paused || this.hidden || video.readyState < 2 || video.videoWidth === 0) {
      schedule();
      return;
    }

    const now = performance.now();
    // MediaPipe rejects a timestamp that does not advance, and two frames can
    // share a millisecond on a fast clock. Forcing monotonicity here is cheaper
    // than losing the frame.
    const timestamp = Math.max(now, this.lastTimestamp + 1);
    this.lastTimestamp = timestamp;

    try {
      const started = performance.now();
      const result = recognizer.recognizeForVideo(video, timestamp);
      const inferenceMs = performance.now() - started;

      const accumulator = this.accumulator;
      accumulator.frames += 1;
      accumulator.inferenceMs += inferenceMs;
      accumulator.seconds += this.lastSampleAt === 0 ? 0 : (now - this.lastSampleAt) / 1000;

      this.consume(result, now);

      if (accumulator.seconds >= 0.5) {
        this.setStatus(this.phase, this.error, {
          fps: accumulator.frames / accumulator.seconds,
          inferenceMs: accumulator.inferenceMs / accumulator.frames,
          delegate: this.delegate,
          handsVisible: this.reading.hands.length,
        });
        accumulator.frames = 0;
        accumulator.seconds = 0;
        accumulator.inferenceMs = 0;
      }
    } catch {
      // A single failed inference is not worth tearing the session down for; it
      // is usually a frame delivered mid-resize. The loop continues and the
      // springs coast through the gap.
    }

    schedule();
  }

  disable(): void {
    this.generation += 1;
    this.stopLoop();

    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    this.stream = null;
    this.video.srcObject = null;

    this.recognizer?.close();
    this.recognizer = null;

    this.banks = { Left: createHandFilterBank(), Right: createHandFilterBank() };
    this.reading = emptyReading();
    this.casts = [];
    this.overlay = emptyOverlay();
    this.lastSampleAt = 0;
    this.lastTimestamp = 0;

    // Decoration falls back to neutral and the springs walk it home over the
    // next second rather than snapping; `engaged` stays true until they arrive,
    // so the consumer keeps applying them.
    //
    // **Pan and zoom are deliberately left where they are.** Under rate control
    // those channels measure how far the operator has flown, not a pose being
    // held, so relaxing them to zero would fly the map back across every metre
    // of that travel the moment the camera is switched off — from the operator's
    // side, turning the rig off would throw their view away.
    //
    // Routed through `resetHandChannels` rather than assigned field by field,
    // because it also clears the spell state — and accumulated charge that
    // nothing clears is an infinite animation frame: `handChannelsSettled`
    // refuses to settle while charge is non-zero, and charge only decays inside
    // the update that a camera-less rig never calls again.
    const channels = this.channels;
    resetHandChannels(channels);
    channels.tracking = false;
    channels.output.frozen = false;
    channels.output.handsVisible = 0;
    channels.output.steerHand = null;
    channels.output.stroke = null;
    channels.output.charge = 0;
    channels.output.chargingHand = null;
    channels.output.roles = { Left: "idle", Right: "idle" };

    this.setStatus("stopped", null, EMPTY_HAND_DIAGNOSTICS);
    this.announce();
    this.scheduleSprings();
  }

  /** Releases everything, including the visibility listener. Not restartable. */
  destroy(): void {
    this.generation += 1;
    this.running = false;
    this.stopSprings();
    this.stopLoop();
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    this.stream = null;
    this.recognizer?.close();
    this.recognizer = null;
    this.subscribers.clear();
    if (this.onVisibility && typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this.onVisibility);
      this.onVisibility = null;
    }
  }
}
