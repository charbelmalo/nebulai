/** The live surface — one canvas, sitting above whatever the stage is showing.
 *
 *  It is deliberately *not* inside `RunDetail`. The five projections the live
 *  view is made of are five meanings of `y` over one shared time axis, and a
 *  surface that only existed while exactly one run was selected could never
 *  show the fleet. So it lives at stage level, always fed by the current
 *  selection, and the mode switch changes what `y` means rather than which
 *  component is mounted.
 *
 *  It sits *beside* `Trajectory`, not instead of it, and the two answer
 *  different questions on purpose. `Trajectory` lays one run out edge to edge:
 *  its track's full width is `t0..t1`, so it always shows everything and never
 *  moves. This shows a window that walks forward with the agent, so the last
 *  ninety seconds keep a readable scale no matter how long the run has been
 *  going. A four-hour run compressed into 600px is a smear; the same run in a
 *  moving window is legible the entire time.
 *
 *  Everything it draws comes from `LiveDriver`. This file is the mount, the
 *  transport chrome and the hover readout — nothing here computes a figure.
 */

import { useEffect, useRef, useState } from "preact/hooks";
import type { RunView } from "../seer/client";
import type { LiveModel } from "../seer/live";
import {
  LiveDriver,
  Y_MODES,
  type LiveHover,
  type LiveRun,
  type YMode,
} from "../scene/seer/LiveDriver";
import { ACTION_COLOR, FIDELITY_TEXTURE, NEUTRAL_INK, stateInk } from "../seer/encoding";

const MODE_TITLE: Record<YMode, string> = {
  score: "y = what kind of work",
  fleet: "y = which run",
  structure: "y = what ran inside what",
};

export function SeerLive(props: { views: RunView[]; live: LiveModel }) {
  const { views, live } = props;
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const driverRef = useRef<LiveDriver | null>(null);
  const [hover, setHover] = useState<LiveHover | null>(null);
  const [following, setFollowing] = useState(true);
  const [windowS, setWindowS] = useState(90);
  const [mode, setMode] = useState<YMode>("score");
  const [dropped, setDropped] = useState(0);

  // The driver reads through this ref rather than closing over `views`, so a
  // refetched snapshot or a changed selection reaches the canvas without
  // tearing down the driver — and without interrupting a morph in flight.
  const latest = useRef({ views, live });
  latest.current = { views, live };

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;
    let disposed = false;

    const driver = new LiveDriver();
    driverRef.current = driver;
    driver.setSource(() => {
      const cur = latest.current;
      const runs: LiveRun[] = cur.views.map((v) => ({
        runId: v.run_id,
        label: v.summary?.label || v.run_id.replace(/^run_/, ""),
        state: v.state,
        view: v,
        openSpans: cur.live.openSpans(v.run_id),
        marks: cur.live.marks(v.run_id),
      }));
      return { runs };
    });
    driver.onHover = setHover;
    driver.onFollowChange = setFollowing;
    driver.onWindowChange = (s) => setWindowS(Math.round(s));
    driver.setReducedMotion(
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    );
    driver.init(canvas);
    setWindowS(Math.round(driver.windowS));

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
      if (disposed) return;
      let n = 0;
      for (const v of latest.current.views) n += latest.current.live.droppedMarks(v.run_id);
      setDropped(n);
    }, 1000);

    // Debug handle, same pattern as `__sessionDriver`: browser verification
    // needs to drive one frame and read state back without a mouse.
    (window as unknown as { __seerLive?: LiveDriver }).__seerLive = driver;

    return () => {
      disposed = true;
      window.clearInterval(poll);
      ro.disconnect();
      driver.dispose();
      driverRef.current = null;
    };
  }, []);

  const pick = (m: YMode) => {
    setMode(m);
    driverRef.current?.setMode(m);
  };

  return (
    <div class="seer-card seer-live">
      <h3>
        Live
        <span class="seer-dim"> {windowS}s window · scroll to zoom, drag to look back</span>
        <span class="seer-live-transport">
          <span class="seer-modes" role="group" aria-label="what the vertical axis means">
            {Y_MODES.map((m) => (
              <button
                key={m}
                type="button"
                class={`seer-btn${mode === m ? " is-on" : ""}`}
                aria-pressed={mode === m}
                onClick={() => pick(m)}
                title={MODE_TITLE[m]}
              >
                {m}
              </button>
            ))}
          </span>
          <button type="button" class="seer-btn" onClick={() => driverRef.current?.fitAll()}>
            fit
          </button>
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

      <div class="seer-live-host" ref={hostRef}>
        <canvas ref={canvasRef} class="seer-live-canvas" />
        {hover && <LiveReadout hover={hover} showRun={views.length > 1} />}
      </div>

      <LiveLegend views={views} mode={mode} />

      <p class="seer-note">
        Bars come from the server's record; anything still running is drawn from
        the event stream and fades out at its leading edge, because it has no
        end to draw yet. A hollow diamond is a call that finished without a
        clock — reconciled history has no per-item timing, and a bar of any
        length there would be a duration we never measured. Switching the axis
        moves the same marks rather than redrawing them: these are one set of
        events grouped three ways.
        {mode === "structure" && <StructureNote views={views} />}
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
function LiveReadout(props: { hover: LiveHover; showRun: boolean }) {
  const h = props.hover;
  const texture = FIDELITY_TEXTURE[h.fidelity];
  return (
    <div
      class="seer-live-readout tnum"
      style={{ translate: `${Math.round(h.x) + 12}px ${Math.round(h.y) + 12}px` }}
    >
      {props.showRun && <span class="seer-dim">{h.runLabel}</span>}
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

/** The legend follows the axis.
 *
 *  In score mode the rows are actions, so the hue legend is what you need. In
 *  fleet mode the rows are runs and the row *names* are inked by state, so the
 *  legend that matters is which state is which colour. The hues never change
 *  meaning — only which of them the reader currently needs spelled out. */
/** What a flat structure means, said out loud.
 *
 *  Depth comes from `parent_span_id` and from nothing else. That field is on
 *  the wire and is null in every capture so far, so structure mode almost
 *  always draws one row of work under each run — and a reader who is not told
 *  why will read that as "this agent did nothing nested" rather than "no
 *  adapter reported nesting". The check below is a presence test on a field,
 *  not a recomputation of a server figure: it asks whether the record contains
 *  the relationship at all, which is a question about capture. */
function StructureNote(props: { views: RunView[] }) {
  const nested = props.views.some((v) => v.spans?.some((s) => s.parent_span_id));
  if (nested) {
    return (
      <>
        {" "}
        Depth is the nesting the agent reported, never nesting we guessed from
        one call sitting inside another's interval. Two rows at the same depth
        mean two calls whose clocks overlapped.
      </>
    );
  }
  return (
    <>
      {" "}
      <span class="seer-warn">
        No nesting was reported for these runs, so every call sits one level
        inside its run.
      </span>{" "}
      That is a fact about the capture, not about the agent — depth comes only
      from <code>parent_span_id</code>, and guessing it from one call falling
      inside another's interval would manufacture the tree the adapter declined
      to report. The pale band on each run's own row is its wall time; the part
      no call covered is the time the record does not account for.
    </>
  );
}

function LiveLegend(props: { views: RunView[]; mode: YMode }) {
  if (props.mode === "structure") {
    const seen = new Map<string, string>();
    for (const v of props.views) seen.set(v.state, v.state.replace(/_/g, " "));
    return (
      <div class="seer-live-legend">
        {[...seen.entries()].map(([state, label]) => (
          <span key={state}>
            <i style={{ background: stateInk(state) }} />
            {label}
          </span>
        ))}
        <span class="seer-dim">
          each run's own row is its wall time; rows below it are what ran inside
          it, one per reported depth and per overlap
        </span>
      </div>
    );
  }
  if (props.mode === "fleet") {
    const seen = new Map<string, string>();
    for (const v of props.views) seen.set(v.state, v.state.replace(/_/g, " "));
    return (
      <div class="seer-live-legend">
        {[...seen.entries()].map(([state, label]) => (
          <span key={state}>
            <i style={{ background: stateInk(state) }} />
            {label}
          </span>
        ))}
        <span class="seer-dim">row names carry the state; bars keep their action hue</span>
      </div>
    );
  }
  return (
    <div class="seer-live-legend">
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
