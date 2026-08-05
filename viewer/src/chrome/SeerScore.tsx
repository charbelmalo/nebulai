/** The Score — the live view's first projection, mounted into a run's detail.
 *
 *  It sits *beside* `Trajectory`, not instead of it, and the two answer
 *  different questions on purpose. `Trajectory` lays the whole run out edge to
 *  edge: its track's full width is `t0..t1`, so it always shows everything and
 *  never moves. The Score shows a window that walks forward with the agent, so
 *  the last ninety seconds keep a readable scale no matter how long the run has
 *  been going. A four-hour run compressed into 600px is a smear; the same run
 *  in a moving window is legible the entire time.
 *
 *  Everything it draws comes from `ScoreDriver`. This file is the mount, the
 *  transport chrome, and the hover readout — nothing here computes a figure.
 */

import { useEffect, useRef, useState } from "preact/hooks";
import type { RunView } from "../seer/client";
import type { LiveModel } from "../seer/live";
import { ScoreDriver, type ScoreHover } from "../scene/seer/ScoreDriver";
import { ACTION_COLOR, FIDELITY_TEXTURE, NEUTRAL_INK } from "../seer/encoding";

export function SeerScore(props: { view: RunView; live: LiveModel }) {
  const { view, live } = props;
  const runId = view.run_id;
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const driverRef = useRef<ScoreDriver | null>(null);
  const [hover, setHover] = useState<ScoreHover | null>(null);
  const [following, setFollowing] = useState(true);
  const [windowS, setWindowS] = useState(90);
  const [dropped, setDropped] = useState(0);

  // The driver reads through this ref rather than closing over `view`, so a
  // refetched snapshot reaches the canvas without tearing down the driver.
  const latest = useRef({ view, live, runId });
  latest.current = { view, live, runId };

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;
    let disposed = false;

    const driver = new ScoreDriver();
    driverRef.current = driver;
    driver.setSource(() => {
      const cur = latest.current;
      return {
        view: cur.view,
        openSpans: cur.live.openSpans(cur.runId),
        marks: cur.live.marks(cur.runId),
      };
    });
    driver.onHover = setHover;
    driver.onFollowChange = setFollowing;
    driver.onWindowChange = (s) => setWindowS(Math.round(s));
    setWindowS(Math.round(driver.windowS));
    driver.setReducedMotion(
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    );
    driver.init(canvas);

    const ro = new ResizeObserver((entries) => {
      if (disposed) return;
      const r = entries[0]?.contentRect;
      if (r) driver.resize(r.width, r.height, window.devicePixelRatio || 1);
    });
    ro.observe(host);
    driver.resize(host.clientWidth, host.clientHeight, window.devicePixelRatio || 1);

    // How much the window is hiding, sampled rather than watched: it only
    // changes when the model trims, and nothing about it needs a frame's
    // latency.
    const poll = window.setInterval(() => {
      if (!disposed) setDropped(latest.current.live.droppedMarks(latest.current.runId));
    }, 1000);

    // Debug handle, same pattern as `__sessionDriver`: browser verification
    // needs to drive one frame and read state back without a mouse.
    (window as unknown as { __seerScore?: ScoreDriver }).__seerScore = driver;

    return () => {
      disposed = true;
      window.clearInterval(poll);
      ro.disconnect();
      driver.dispose();
      driverRef.current = null;
    };
  }, []);

  return (
    <div class="seer-card seer-score">
      <h3>
        Score
        <span class="seer-dim"> {windowS}s window · scroll to zoom, drag to look back</span>
        <span class="seer-score-transport">
          <button
            type="button"
            class={`seer-btn${following ? " is-on" : ""}`}
            onClick={() => driverRef.current?.setFollow(!following)}
            title={
              following
                ? "following the live edge — drag the chart to look back"
                : "parked in the past; click to return to the live edge"
            }
          >
            {following ? "live" : "resume"}
          </button>
        </span>
      </h3>

      <div class="seer-score-host" ref={hostRef}>
        <canvas ref={canvasRef} class="seer-score-canvas" />
        {hover && <ScoreReadout hover={hover} />}
      </div>

      <ScoreLegend />

      <p class="seer-note">
        Bars come from the server's record; anything still running is drawn from
        the event stream and fades out at its leading edge, because it has no
        end to draw yet. A hollow diamond is a call that finished without a
        clock — reconciled history has no per-item timing, and a bar of any
        length there would be a duration we never measured.
        {dropped > 0 && (
          <>
            {" "}
            <span class="seer-warn">
              {dropped.toLocaleString()} older event{dropped === 1 ? "" : "s"} have
              left the window.
            </span>{" "}
            The full log is on disk; this view is bounded on purpose.
          </>
        )}
      </p>
    </div>
  );
}

/** What is under the cursor. Prints a duration only when there is one — an open
 *  span and an unclocked one both say so in words instead. */
function ScoreReadout(props: { hover: ScoreHover }) {
  const h = props.hover;
  const texture = FIDELITY_TEXTURE[h.fidelity];
  return (
    <div
      class="seer-score-readout tnum"
      style={{ translate: `${Math.round(h.x) + 12}px ${Math.round(h.y) + 12}px` }}
    >
      <span style={{ color: h.action ? ACTION_COLOR[h.action] : NEUTRAL_INK }}>
        {h.action ?? "unclassified"}
      </span>
      <strong>
        {h.endedAt == null
          ? "running…"
          : h.durationS == null
            ? "not timed"
            : `${h.durationS.toFixed(h.durationS < 1 ? 2 : 1)}s`}
      </strong>
      {h.detail && <span class="seer-dim">{h.detail}</span>}
      {texture !== "solid" && <span class="seer-dim">{h.fidelity.replace(/_/g, " ")}</span>}
    </div>
  );
}

/** The hue legend, and the two textures that mean "no number here".
 *
 *  Present because the encoding is shared across every projection: someone who
 *  learns it once on the Score reads the field, the fleet strip and the span
 *  tree for free. */
function ScoreLegend() {
  return (
    <div class="seer-score-legend">
      {(Object.keys(ACTION_COLOR) as (keyof typeof ACTION_COLOR)[]).map((a) => (
        <span key={a}>
          <i style={{ background: ACTION_COLOR[a] }} />
          {a}
        </span>
      ))}
      <span title="the agent never reported it — an unfilled shape, never a short one">
        <i class="is-outline" />
        missing
      </span>
      <span title="we chose not to capture it">
        <i class="is-policy" />
        policy
      </span>
    </div>
  );
}
