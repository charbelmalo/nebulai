/** SeerSample — what a captured run looks like, shown while the collector is
 *  offline.
 *
 *  Without `seer serve` the Live page had nothing on its stage but prose and a
 *  rail of forms that could not submit. This puts one run's shape in front of
 *  the visitor instead: action lanes over time, in the same ACTION_COLOR the
 *  real trajectory uses, so the first real run reads as familiar.
 *
 *  ## What it must never be mistaken for
 *
 *  Every number and bar here is a fixed, hand-written illustration bundled with
 *  the app. It is labelled "Sample run — not live" at the top, is never mixed
 *  into the run list, never selectable, never exportable, and disappears the
 *  moment the collector answers. Nothing in it was measured on this machine. */

import { useSignal } from "@preact/signals";
import type { Action } from "../seer/contract";
import { ACTION_COLOR } from "../seer/encoding";

/** [action, start s, end s] — one plausible 3-minute fix-and-verify run. */
const SAMPLE_SPANS: [Action, number, number][] = [
  ["inspect", 0, 9],
  ["search", 9, 15],
  ["inspect", 15, 27],
  ["search", 27, 31],
  ["inspect", 31, 44],
  ["edit", 44, 58],
  ["execute", 58, 79],
  ["inspect", 79, 86],
  ["edit", 86, 97],
  ["execute", 97, 112],
  ["verify", 112, 131],
  ["edit", 131, 138],
  ["verify", 138, 157],
  ["vcs", 157, 166],
  ["verify", 166, 178],
  ["report", 178, 192],
];
const SAMPLE_WALL = 192;
const SAMPLE_LANES: Action[] = ["inspect", "search", "edit", "execute", "verify", "vcs", "report"];

const START_COMMAND = "seer serve";

export function SeerSample() {
  const copied = useSignal<"idle" | "ok" | "fail">("idle");
  const pct = (t: number) => (t / SAMPLE_WALL) * 100;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(START_COMMAND);
      copied.value = "ok";
    } catch {
      copied.value = "fail";
    }
  };

  return (
    <section class="seer-sample" aria-labelledby="seer-sample-title">
      <header class="seer-sample-head">
        <span class="seer-sample-badge">Sample run — not live</span>
        <h3 id="seer-sample-title">
          claude <span class="seer-sample-dim">· completed · 3m 12s · 41 events</span>
        </h3>
      </header>
      {/* HTML rather than SVG so the lane labels hold 12px at every width; an
          SVG viewBox shrank them to ~6px on a phone. */}
      <div
        class="seer-sample-lanes"
        role="img"
        aria-label="Illustrative sample: action lanes over a 3 minute 12 second run. Not measured data."
      >
        {SAMPLE_LANES.map((a) => (
          <div key={a} class="seer-sample-row">
            <span class="seer-sample-lane">{a}</span>
            <span class="seer-sample-track">
              {SAMPLE_SPANS.filter(([sa]) => sa === a).map(([, t0, t1], k) => (
                <i
                  key={k}
                  style={{
                    left: `${pct(t0)}%`,
                    width: `calc(${pct(t1) - pct(t0)}% - 1.5px)`,
                    background: ACTION_COLOR[a],
                  }}
                />
              ))}
            </span>
          </div>
        ))}
        <div class="seer-sample-row seer-sample-axis" aria-hidden="true">
          <span />
          <span class="seer-sample-track">
            {[0, 60, 120, 180].map((t) => (
              <b key={t} style={{ left: `${pct(t)}%` }}>
                {t === 0 ? "0" : `${t / 60}m`}
              </b>
            ))}
          </span>
        </div>
      </div>
      <p class="seer-sample-caption">
        Each bar is one tool call on its action's lane; the run warms from reading to editing to
        verifying. An illustration bundled with this page — nothing here was measured.
      </p>
      <div class="seer-sample-cta">
        <button type="button" class="btn-primary" onClick={copy}>
          Copy <code>{START_COMMAND}</code>
        </button>
        <span class="seer-sample-cta-note" role="status">
          {copied.value === "ok"
            ? "Copied. Run it in a terminal — this page connects on its own."
            : copied.value === "fail"
              ? `Copy failed — type ${START_COMMAND} in a terminal; this page connects on its own.`
              : "Start the collector, and live runs replace this sample."}
        </span>
      </div>
    </section>
  );
}

/** The rail while the collector is down: what the launcher and importer do,
 *  without a form that cannot submit. */
export function SeerRailOffline() {
  return (
    <section class="seer-launch seer-rail-offline">
      <h3>Capture or import runs</h3>
      <p class="seer-note">
        Launch Codex, Claude Code or Hermes headless, or import past Codex sessions — both need the
        local collector. Start it with <code>{START_COMMAND}</code>; the forms appear once it
        answers.
      </p>
    </section>
  );
}
