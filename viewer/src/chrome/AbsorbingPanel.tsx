/** AbsorbingPanel — the Waluigi absorbing-state readout (Attractors P3).
 *
 *  "A character that slips out of persona tends to stay slipped" is the single
 *  most repeated claim about conversational LLMs that nobody has measured
 *  cleanly on open weights. `nebulai absorbing run` measures it: a pinned
 *  instruct model plays both sides of a conversation under a persona rule,
 *  every assistant turn is judged by a STATED DETERMINISTIC REGEX, and the
 *  transition matrix is counted.
 *
 *  ## What this panel refuses to do
 *
 *  It never renders P(violate | violated) on its own. The conditional is
 *  drawn on the same axis as the base rate, with its Wilson interval, above the
 *  within-conversation shuffle null — because a conditional without a base rate
 *  is a number that cannot be wrong, and one without the null is a number that
 *  cannot tell an absorbing state from a set of conversations that simply
 *  differ from each other (R5: the null ships with the figure).
 *
 *  It never shows the judge as a summary. `rule.judge` prints verbatim, and so
 *  does the regex, because the whole result is a statement about that rule.
 *
 *  And it never shows an N without saying whether the run finished. A study
 *  stopped by a deadline reports the N it reached, labelled — a short run that
 *  prints its number the same way a complete one does is exactly the shape of a
 *  result that looks more finished than it is.
 *
 *  The arithmetic lives in `data/absorbing.ts` and is unit-tested there; this
 *  file is layout and words. `intervalDisagreement()` is rendered when it fires
 *  rather than swallowed: a study file whose intervals do not match the formula
 *  is worth seeing, and the panel is the only place anybody would see it. */

import { useEffect } from "preact/hooks";
import { signal } from "@preact/signals";
import {
  intervalDisagreement,
  loadStudy,
  loadStudyIndex,
  spans,
  verdictNote,
  type AbsorbingStudy,
  type Rate,
} from "../data/absorbing";

/** Available study ids, or null until the index has been read. The three-state
 *  (null / [] / ids) is the point: "not looked yet" and "looked, none shipped"
 *  must not render the same way. */
const $ids = signal<string[] | null>(null);
const $studyId = signal<string | null>(null);
const $study = signal<AbsorbingStudy | null>(null);
const $error = signal<string | null>(null);
const $collapsed = signal(false);

let indexRequested = false;

function ensureIndex(): void {
  if (indexRequested) return;
  indexRequested = true;
  loadStudyIndex()
    .then((list) => {
      $ids.value = list.map((s) => s.studyId);
      if (!$studyId.value && list.length) $studyId.value = list[0]!.studyId;
    })
    .catch(() => ($ids.value = []));
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function fmtInterval(iv: [number, number] | null): string {
  return iv === null ? "no interval — n = 0" : `${pct(iv[0])} – ${pct(iv[1])}`;
}

/** One rate as a labelled bar with its interval drawn as the bar, not as a
 *  caption. The interval IS the measurement; a point marker inside it is the
 *  estimate. Drawing it the other way round — a bar to `p` with whiskers — puts
 *  the visual weight on the number least entitled to it. */
function RateBar(props: {
  label: string;
  title: string;
  rate: Rate;
  accent?: boolean;
  /** value to mark on the same axis for comparison (the base rate) */
  marker?: number | null;
}) {
  const r = props.rate;
  const iv = r.ci95;
  const lo = iv ? iv[0] : r.p;
  const hi = iv ? iv[1] : r.p;
  return (
    <div class={`absorb-rate${props.accent ? " absorb-rate-accent" : ""}`} title={props.title}>
      <div class="absorb-rate-head">
        <span class="absorb-rate-label">{props.label}</span>
        <span class="absorb-rate-n">
          {r.k}/{r.n}
        </span>
      </div>
      <div class="absorb-axis" aria-hidden="true">
        {iv === null ? (
          <div class="absorb-noband" />
        ) : (
          <div
            class="absorb-band"
            style={{ left: `${lo * 100}%`, width: `${Math.max(0.6, (hi - lo) * 100)}%` }}
          />
        )}
        <div class="absorb-point" style={{ left: `${r.p * 100}%` }} />
        {typeof props.marker === "number" && (
          <div class="absorb-marker" style={{ left: `${props.marker * 100}%` }} />
        )}
      </div>
      <p class="absorb-rate-foot">
        <b>{r.n > 0 ? pct(r.p) : "not measured"}</b>
        <span class="absorb-ci">95% {fmtInterval(iv)}</span>
      </p>
    </div>
  );
}

function TransitionMatrix(props: { study: AbsorbingStudy }) {
  const c = props.study.counts;
  const total = c.n00 + c.n01 + c.n10 + c.n11;
  const cell = (v: number) => (
    <td>
      <b>{v.toLocaleString()}</b>
      <span class="absorb-cell-share">{total > 0 ? pct(v / total) : "—"}</span>
    </td>
  );
  return (
    <table class="absorb-matrix">
      <caption>
        {total.toLocaleString()} transitions — rows are the state at turn <i>t</i>, columns the
        state at <i>t</i>+1
      </caption>
      <thead>
        <tr>
          <th scope="col" />
          <th scope="col">→ in character</th>
          <th scope="col">→ violation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">in character</th>
          {cell(c.n00)}
          {cell(c.n01)}
        </tr>
        <tr>
          <th scope="row">violation</th>
          {cell(c.n10)}
          {cell(c.n11)}
        </tr>
      </tbody>
    </table>
  );
}

export function AbsorbingPanel() {
  ensureIndex();
  const ids = $ids.value;
  const id = $studyId.value;

  useEffect(() => {
    if (!id) return;
    let live = true;
    $error.value = null;
    loadStudy(id)
      .then((s) => {
        if (live) $study.value = s;
      })
      .catch((e) => {
        if (live) {
          $study.value = null;
          $error.value = String(e?.message ?? e);
        }
      });
    return () => {
      live = false;
    };
  }, [id]);

  // nothing shipped: render nothing at all rather than an empty card. The
  // study is optional, and an "absent" placeholder on a page that never had
  // one is noise, not honesty.
  if (ids !== null && ids.length === 0) return null;
  if (ids === null) return null;

  if ($collapsed.value) {
    return (
      <button
        type="button"
        class="absorb-fab"
        aria-label="Open the absorbing-state readout"
        onClick={() => ($collapsed.value = false)}
      >
        Absorbing state
      </button>
    );
  }

  const s = $study.value;

  return (
    <section class="absorb" aria-label="Absorbing-state study">
      <header class="absorb-head">
        <h2 class="absorb-title">Does a slipped character stay slipped?</h2>
        <button
          type="button"
          class="absorb-collapse"
          aria-label="Collapse the absorbing-state readout"
          onClick={() => ($collapsed.value = true)}
        >
          ›
        </button>
      </header>

      {ids.length > 1 && (
        <label class="absorb-pick">
          <span>Study</span>
          <select
            value={id ?? ""}
            onChange={(e) => ($studyId.value = (e.target as HTMLSelectElement).value)}
          >
            {ids.map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </label>
      )}

      {$error.value && <p class="absorb-error">{$error.value}</p>}

      {s && (
        <>
          <p class="absorb-prov">
            <b>{s.model}</b> @ <code>{s.revision.slice(0, 12)}</code> · rule{" "}
            <code>{s.rule.id}</code> · {s.nConversations.toLocaleString()} conversations ×{" "}
            {s.nTurns} turns = {s.nJudgedTurns.toLocaleString()} judged turns
          </p>

          {s.stoppedEarly && (
            <p class="absorb-warn">
              Stopped early at {s.nConversations.toLocaleString()} of the{" "}
              {s.nConversationsRequested.toLocaleString()} conversations requested
              {s.deadlineS !== null && ` (deadline ${(s.deadlineS / 60).toFixed(0)} min)`}. Every
              number below is over the N actually reached.
            </p>
          )}

          <p class="absorb-rule">
            <b>The rule.</b> {s.rule.statement} Judged by <code>/{s.rule.pattern}/</code>.
          </p>
          <p class="absorb-judge">{s.rule.judge}</p>

          <div class="absorb-sep" />

          <RateBar
            label="P(violate at t+1 | violated at t)"
            title="the conditional the absorbing-state claim is about"
            rate={s.pGivenViolated}
            accent
            marker={s.baseRate.n > 0 ? s.baseRate.p : null}
          />
          <RateBar
            label="P(violate at t+1 | in character at t)"
            title="the other row of the matrix, for contrast"
            rate={s.pGivenInCharacter}
            marker={s.baseRate.n > 0 ? s.baseRate.p : null}
          />
          <RateBar
            label="base rate"
            title={s.baseRate.over || "every turn that can be a t+1"}
            rate={s.baseRate}
          />

          <p class={`absorb-span ${s.intervalSpansBaseRate ? "absorb-span-null" : "absorb-span-clear"}`}>
            {spans(s.pGivenViolated.ci95, s.baseRate.p)
              ? "The conditional's 95% interval SPANS the base rate — at this N the two are not distinguishable."
              : "The conditional's 95% interval excludes the base rate."}
          </p>

          <div class="absorb-sep" />
          <TransitionMatrix study={s} />

          <div class="absorb-sep" />
          <h3 class="absorb-subtitle">The null</h3>
          <p class="absorb-null">
            <span>
              {s.null.method.replace(/_/g, " ")}, n = {s.null.n.toLocaleString()}, on{" "}
              {s.null.statistic.replace(/_/g, " ")}
            </span>
            <span>
              p95 <b>{s.null.p95 === null ? "not measured" : pct(s.null.p95)}</b>
              {s.null.mean !== null && <> (mean {pct(s.null.mean)})</>}
              {s.null.pValue !== null && (
                <>
                  {" "}
                  · p <b>{s.null.pValue.toFixed(4)}</b>
                </>
              )}
            </span>
          </p>
          {s.null.note && <p class="absorb-caption">{s.null.note}</p>}

          <div class="absorb-sep" />
          <p class={`absorb-verdict absorb-verdict-${s.verdict}`}>
            <b>{s.verdict.replace(/_/g, " ")}</b>
          </p>
          <p class="absorb-caption">{verdictNote(s)}</p>

          {s.pilot && (
            <p class="absorb-caption">
              The rule was chosen by a pilot BEFORE any transition was counted — violation rates{" "}
              {Object.entries(s.pilot.rates)
                .map(([k, v]) => `${k} ${(v as number).toFixed(3)}`)
                .join(", ")}{" "}
              — so the choice could not be tuned to the result.
            </p>
          )}

          {intervalDisagreement(s).map((line) => (
            <p class="absorb-error" key={line}>
              {line}
            </p>
          ))}
        </>
      )}
    </section>
  );
}
