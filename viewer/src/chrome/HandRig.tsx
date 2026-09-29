/** HandRig.tsx — the operator console for webcam hand control.
 *
 *  Three jobs, in order of importance:
 *
 *  1. **Show that the camera is on and what it sees.** A page holding a webcam
 *     stream owes the person in front of it an unambiguous, always-visible
 *     picture of the feed. The preview is not optional chrome and there is no
 *     setting to hide it while tracking runs — turning the panel off means
 *     turning the camera off.
 *  2. **Explain a gesture that did not fire.** The skeleton's colour is the
 *     role the channel layer assigned, the ring is the palm centre every
 *     velocity is measured from, and the arc is the charge a shockwave will
 *     spend. When something will not trigger, the answer is nearly always one
 *     of those three.
 *  3. **Teach the vocabulary.** Short enough now to print in full — three hand
 *     shapes and five outcomes — so the legend is part of the panel rather than
 *     documentation. It was fifteen outcomes across nine shapes and three
 *     selectable modes, which is what made a legend feel like a manual.
 *
 *  On what does *not* re-render: the charge meter is written straight to the
 *  element's `transform` from the tracker's subscription. Charge changes every
 *  frame while a palm is held still, and routing that through a signal would
 *  re-render this whole subtree at display cadence for one number. Preact is
 *  fast, but "fast enough to redraw a panel 60 times a second" is not the same
 *  claim as "should". The same applies to the canvas, which is drawn imperatively
 *  from the same subscription.
 */

import { useSignal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";
import { appStore } from "../app/store";
import { drawHandOverlay } from "./handOverlay";
import { handRig, type HandRigState } from "../hands/rig";
import {
  HAND_EFFECTS_LEGEND,
  HAND_LEGEND,
  SPELL_LABEL,
} from "../hands/types";
import { $settings } from "./state";

function phaseLabel(state: HandRigState): string {
  switch (state.phase) {
    case "idle":
      return "starting…";
    case "requesting":
      return "waiting for camera permission…";
    case "loading":
      return "loading the hand-tracking model…";
    case "tracking":
      return state.diagnostics.handsVisible > 0
        ? `${state.diagnostics.handsVisible} hand${state.diagnostics.handsVisible > 1 ? "s" : ""}`
        : "no hands in frame";
    case "stopped":
      return "camera released";
    case "error":
      return state.error ?? "hand control failed";
  }
}

/**
 * The gate is its own component so the panel below can use hooks.
 *
 * Preact matches hooks by call order, and a hook after an early return changes
 * that order the moment the setting flips — which is exactly when this component
 * re-renders. Splitting the condition out means the panel either mounts whole or
 * does not exist, and unmounting it is also what tears down the subscription and
 * the canvas.
 */
export function HandRig() {
  return $settings.value.handTracking ? <HandRigPanel /> : null;
}

function HandRigPanel() {
  const state = useSignal<HandRigState>(handRig.snapshot);
  const legendOpen = useSignal(false);
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const meter = useRef<HTMLSpanElement>(null);

  useEffect(() => handRig.subscribe((next) => (state.value = next)), []);

  const phase = state.value.phase;

  // Adopt the tracker's own <video>. The element belongs to the tracker rather
  // than to this component on purpose: the rig outlives the panel — it keeps
  // integrating with the Settings overlay open over the top — and a preview the
  // tracker had to wait for would make the camera's lifetime depend on a render
  // order nobody controls.
  useEffect(() => {
    const video = handRig.session?.video;
    const container = host.current;
    if (!video || !container || video.parentElement === container) return;
    video.className = "hand-rig-video";
    container.prepend(video);
  }, [phase]);

  // Overlay + charge meter, both written imperatively from the tracker's own
  // announcements. `subscribe` returns its unsubscribe, so the effect's cleanup
  // is the subscription's.
  useEffect(() => {
    const tracker = handRig.session;
    const element = canvas.current;
    if (!tracker || !element) return;
    const ctx = element.getContext("2d");
    if (!ctx) return;

    const reducedMotion =
      appStore.getState().settings.reducedMotion ||
      (typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches);

    const paint = (): void => {
      drawHandOverlay(element, ctx, tracker.overlay, { reducedMotion });
      const fill = meter.current;
      if (fill) fill.style.transform = `scaleX(${tracker.overlay.charge})`;
    };

    paint();
    return tracker.subscribe(paint);
  }, [phase]);

  const cast = state.value.cast;
  const diagnostics = state.value.diagnostics;

  return (
    <section class={`hand-rig${phase === "error" ? " is-error" : ""}`} aria-label="Hand control">
      <div class="hand-rig-stage" ref={host}>
        <canvas class="hand-rig-canvas" ref={canvas} aria-hidden="true" />
        <span class="hand-rig-meter" aria-hidden="true">
          <span class="hand-rig-meter-fill" ref={meter} />
        </span>
      </div>

      <header class="hand-rig-head">
        <span class={`hand-rig-dot is-${phase}`} aria-hidden="true" />
        <span class="hand-rig-phase">{phaseLabel(state.value)}</span>
        <button
          type="button"
          class="hand-rig-stop"
          onClick={() => appStore.getState().setSetting("handTracking", false)}
        >
          Turn off
        </button>
      </header>

      {phase === "tracking" && (
        <p class="hand-rig-diag">
          {diagnostics.fps.toFixed(0)} fps · {diagnostics.inferenceMs.toFixed(1)} ms ·{" "}
          {diagnostics.delegate}
        </p>
      )}

      {/* The one place a cast is announced to a screen reader. Polite, not
          assertive: a spell is a thing the operator just did on purpose, so it
          must not interrupt whatever they were being read. */}
      <p class="hand-rig-live" aria-live="polite">
        {cast ? `${SPELL_LABEL[cast.id]} cast` : ""}
      </p>

      <button
        type="button"
        class="hand-rig-legend-toggle"
        aria-expanded={legendOpen.value}
        onClick={() => (legendOpen.value = !legendOpen.value)}
      >
        {legendOpen.value ? "Hide gestures" : "Show gestures"}
      </button>

      {legendOpen.value && (
        <div class="hand-rig-legend">
          {/* One fixed list, because there is now one control law. It used to be
              keyed on the selected mode, which was the honest thing to do while
              three modes existed — a legend describing gestures the active mode
              does not implement makes an operator conclude the camera is broken
              rather than that they are reading the wrong page. Deleting the
              modes is what let the list become a constant. */}
          <h3 class="hand-rig-legend-title">Gestures</h3>
          <dl class="hand-rig-legend-list">
            {HAND_LEGEND.map((entry) => (
              <div class="hand-rig-legend-row" key={entry.pose}>
                <dt>{entry.pose}</dt>
                <dd>{entry.effect}</dd>
              </div>
            ))}
          </dl>
          {/* Same rule, applied to the optional half: with effects off these
              three gestures do nothing at all, so listing them unconditionally
              would be exactly the instructions-that-do-nothing failure above. */}
          {$settings.value.handEffects && (
            <>
              <h3 class="hand-rig-legend-title">Effects</h3>
              <dl class="hand-rig-legend-list">
                {HAND_EFFECTS_LEGEND.map((entry) => (
                  <div class="hand-rig-legend-row" key={entry.pose}>
                    <dt>{entry.pose}</dt>
                    <dd>{entry.effect}</dd>
                  </div>
                ))}
              </dl>
            </>
          )}
          <p class="hand-rig-note">
            Video is processed on this machine. No frame is uploaded, stored or sent anywhere.
          </p>
        </div>
      )}
    </section>
  );
}
