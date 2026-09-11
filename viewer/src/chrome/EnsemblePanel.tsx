/** EnsemblePanel — N runs of one protocol, read as a sample (Attractors P3).
 *
 *  One run of an agent is an anecdote. `seer run … --repeat N` makes it a
 *  sample, and this panel is what a sample is for: a band rather than a line,
 *  rates with intervals rather than points, and a reliability number that says
 *  whether two conditions differ by more than each differs from itself.
 *
 *  ## The rule this component ENFORCES rather than documents
 *
 *  Below the document's own `point_estimate_min_runs` there is no point
 *  estimate on screen. Not a number with a caveat beside it, not a number in a
 *  lighter weight — an interval and the raw `k/n`. The rule lives in
 *  `RateRow`, which has no branch that can print a percentage for an
 *  under-powered rate: `pointEstimateAllowed()` gates the only place `pct(p)`
 *  appears. That is deliberate. A rule enforced by the caller is a rule that
 *  holds until somebody adds a second caller, and a reader quotes the number
 *  they can read off the screen no matter what the footnote says.
 *
 *  The threshold is read out of the DOCUMENT, never from a constant here, so
 *  the panel and the backend cannot drift to two different numbers.
 *
 *  ## What else it refuses
 *
 *  - A band is never drawn without its per-step `n`. Runs that ended early are
 *    absent from the steps they never reached, so a fan narrows towards the
 *    right for a reason that has nothing to do with the agent agreeing with
 *    itself. The attrition line says so in words whenever it happens, and names
 *    the step where the sample first thinned.
 *  - A quantity the backend could not compute is printed with its reason. It is
 *    never zero and never blank.
 *  - Δ̂ is shown with its three terms and its sign. A Δ̂ near zero because two
 *    conditions are alike and a Δ̂ near zero because both are internally noisy
 *    are different findings, and only the terms tell them apart.
 *  - No verdict is invented. §6.4.1 gives no threshold at which Δ̂ becomes
 *    "significant" and the panel does not legislate one.
 *
 *  The arithmetic is in `data/ensemble.ts` and unit-tested there against a real
 *  100-run document; this file is layout and words. */

import { useEffect } from "preact/hooks";
import { signal } from "@preact/signals";
import {
  envelopeWidths,
  intervalDisagreement,
  missingLines,
  parseEnsemble,
  pointEstimateAllowed,
  rateInterval,
  reliabilityNote,
  shortfall,
  survivorship,
  underpoweredNote,
  type Ensemble,
  type EnsembleRate,
} from "../data/ensemble";
import { fetchEnsemble, fetchEnsembles } from "../seer/client";
import { appStore } from "../app/store";

/** Available ensemble ids, or null until the list has been read. The three
 *  states are the point: "not looked yet" and "looked, none exist" must not
 *  render the same way. */
const $ids = signal<string[] | null>(null);
const $ensemble = signal<Ensemble | null>(null);
const $error = signal<string | null>(null);
const $collapsed = signal(false);

let indexRequested = false;

function ensureIndex(): void {
  if (indexRequested) return;
  indexRequested = true;
  fetchEnsembles()
    .then((list) => {
      $ids.value = list.map((e) => e.ensembleId);
      const cur = appStore.getState().sessions.ensembleId;
      if (!cur && list.length) appStore.getState().setEnsemble(list[0]!.ensembleId);
    })
    .catch(() => ($ids.value = []));
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function fmtInterval(iv: [number, number] | null): string {
  return iv === null ? "no interval — nothing was measured" : `${pct(iv[0])} – ${pct(iv[1])}`;
}

function num(v: number | null, digits = 4): string {
  return v === null ? "—" : v.toFixed(digits);
}

/** The label each rate key gets. Spelled out rather than prettified from the
 *  key, because "verified_after_last_edit" needs a sentence, not title case. */
const RATE_LABELS: Record<string, string> = {
  completed: "reached a terminal state",
  failed: "ended in failure",
  verified: "ran any verification",
  edited: "performed at least one edit",
  files_changed: "touched a named file",
  verified_after_last_edit: "verified after the last edit",
};

/** One rate. The interval IS the bar; a point marker inside it is the estimate,
 *  and it is drawn ONLY when the sample supports one. */
function RateRow(props: { e: Ensemble; rate: EnsembleRate }) {
  const { e, rate } = props;
  const iv = rateInterval(rate);
  const allowed = pointEstimateAllowed(e, rate);
  const label = RATE_LABELS[rate.key] ?? rate.key;

  if (rate.fidelity === "missing" || iv === null) {
    return (
      <div class="ens-rate ens-rate-missing">
        <div class="ens-rate-head">
          <span class="ens-rate-label">{label}</span>
          <span class="ens-rate-value ens-missing">not measured</span>
        </div>
        <div class="ens-rate-track">
          <div class="ens-noband" title="nothing was measured" />
        </div>
        {rate.missing ? <p class="ens-note ens-note-missing">{rate.missing}</p> : null}
      </div>
    );
  }

  const lo = iv[0];
  const hi = iv[1];
  return (
    <div class={`ens-rate${allowed ? "" : " ens-rate-underpowered"}`}>
      <div class="ens-rate-head">
        <span class="ens-rate-label">{label}</span>
        {/* the ONLY place a percentage is printed, and it is gated */}
        <span class="ens-rate-value">
          {allowed ? pct(rate.p as number) : `${rate.k}/${rate.n}`}
        </span>
      </div>
      <div class="ens-rate-track">
        <div
          class="ens-band"
          style={{ left: `${lo * 100}%`, width: `${Math.max(0.6, (hi - lo) * 100)}%` }}
          title={fmtInterval(iv)}
        />
        {allowed ? (
          <div class="ens-tick" style={{ left: `${(rate.p as number) * 100}%` }} />
        ) : null}
      </div>
      <div class="ens-rate-foot">
        <span class="ens-mono">{fmtInterval(iv)}</span>
        <span class="ens-mono ens-dim">
          {rate.k}/{rate.n}
        </span>
      </div>
      {allowed ? null : <p class="ens-note ens-note-warn">{underpoweredNote(e, rate)}</p>}
      {rate.note ? <p class="ens-note">{rate.note}</p> : null}
      {rate.nUndecidable ? (
        <p class="ens-note">
          {rate.nUndecidable} run{rate.nUndecidable === 1 ? "" : "s"} could not answer this
          question and {rate.nUndecidable === 1 ? "is" : "are"} excluded from n — never counted
          as a failure.
        </p>
      ) : null}
    </div>
  );
}

/** The fan, as an SVG. The band is the drawing; the median is a line inside it,
 *  and the per-step n is its own strip underneath so the reader sees the sample
 *  thinning at the same x where the band narrows. */
function Fan(props: { e: Ensemble }) {
  const e = props.e;
  if (!e.fan.length) {
    return (
      <p class="ens-note ens-note-missing">
        {e.missing["fan"] ?? "no fan in this document, and no reason given for its absence"}
      </p>
    );
  }
  const W = 520;
  const H = 150;
  const NH = 18;
  const steps = e.fan;
  const maxStep = Math.max(1, steps[steps.length - 1]!.step);
  const maxY = Math.max(...steps.map((s) => s.hi)) || 1;
  const nMax = Math.max(...steps.map((s) => s.n)) || 1;
  const x = (s: number) => (s / maxStep) * W;
  const y = (v: number) => H - (v / maxY) * H;

  const upper = steps.map((s) => `${x(s.step)},${y(s.hi)}`).join(" ");
  const lower = [...steps].reverse().map((s) => `${x(s.step)},${y(s.lo)}`).join(" ");
  const median = steps.map((s) => `${x(s.step)},${y(s.median)}`).join(" ");
  const surv = survivorship(e);

  return (
    <div class="ens-fan">
      <svg viewBox={`0 0 ${W} ${H + NH + 6}`} class="ens-fan-svg" role="img">
        <polygon class="ens-fan-band" points={`${upper} ${lower}`} />
        <polyline class="ens-fan-median" points={median} />
        {/* the per-step n, as its own strip: the band narrowing and the sample
            shrinking are two different facts and must be separable by eye */}
        {steps.map((s, i) => {
          const w = i + 1 < steps.length ? x(steps[i + 1]!.step) - x(s.step) : 2;
          return (
            <rect
              class="ens-fan-n"
              x={x(s.step)}
              y={H + 6}
              width={Math.max(1, w)}
              height={(s.n / nMax) * NH}
              opacity={0.25 + 0.75 * (s.n / nMax)}
            />
          );
        })}
      </svg>
      <p class="ens-note">
        Band: {e.fanEnvelope.replace(/_/g, "/")} of {e.fanMetric.replace(/_/g, " ")} over the
        runs that reached each step. Read the WIDTH, not the line — the median is a summary of
        the band, not a result on its own.
      </p>
      {surv.narrowsByAttrition ? (
        <p class="ens-note ens-note-warn">
          The sample thins from {surv.nAtStart} runs to {surv.nAtEnd} by the last step
          {surv.firstDropStep === null ? "" : `, first dropping at step ${surv.firstDropStep}`}.
          The band narrows on the right partly because fewer runs are in it, which is
          attrition, not agreement.
        </p>
      ) : (
        <p class="ens-note">Every run reached every step: no attrition narrows this band.</p>
      )}
      <p class="ens-note ens-dim">
        Widest band: {Math.max(...envelopeWidths(e).map((w) => w.width)).toFixed(1)}{" "}
        {e.fanMetric.replace(/_/g, " ")}.
      </p>
    </div>
  );
}

function Reliability(props: { e: Ensemble }) {
  const r = props.e.reliability;
  return (
    <section class="ens-block">
      <h4 class="ens-h">
        Δ̂ — does the contrast exceed each condition's own noise?
      </h4>
      {r.conditionA && r.conditionB ? (
        <p class="ens-note ens-mono">
          {r.conditionA} (n = {r.nA ?? "?"}) vs {r.conditionB} (n = {r.nB ?? "?"})
        </p>
      ) : null}
      <div class="ens-terms">
        <div>
          <span class="ens-term-label">Δ̂</span>
          <span class="ens-term-value">{num(r.deltaHat)}</span>
        </div>
        <div>
          <span class="ens-term-label">between</span>
          <span class="ens-term-value">{num(r.between)}</span>
        </div>
        <div>
          <span class="ens-term-label">within {r.conditionA ?? "A"}</span>
          <span class="ens-term-value">{num(r.withinA)}</span>
        </div>
        <div>
          <span class="ens-term-label">within {r.conditionB ?? "B"}</span>
          <span class="ens-term-value">{num(r.withinB)}</span>
        </div>
      </div>
      <p class="ens-note">{reliabilityNote(r)}</p>
      {r.formula ? <p class="ens-note ens-mono ens-dim">{r.formula}</p> : null}
      {r.source ? <p class="ens-note ens-dim">{r.source}</p> : null}
      {r.featuresUsed.length ? (
        <p class="ens-note ens-dim">
          Measured on: {r.featuresUsed.join(", ")}
          {r.kernel ? ` · ${r.kernel} kernel, ${r.estimator}` : ""}
          {r.bandwidth === null ? "" : `, bandwidth ${r.bandwidth.toFixed(3)}`}
        </p>
      ) : null}
      {Object.keys(r.featuresDropped).length ? (
        <ul class="ens-dropped">
          {Object.entries(r.featuresDropped).map(([f, why]) => (
            <li key={f}>
              <span class="ens-mono">{f}</span> — dropped for every run: {why}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

export function EnsemblePanel() {
  const ensembleId = appStore.getState().sessions.ensembleId;

  useEffect(() => {
    ensureIndex();
  }, []);

  useEffect(() => {
    if (!ensembleId) {
      $ensemble.value = null;
      return;
    }
    let cancelled = false;
    fetchEnsemble(ensembleId)
      .then((doc) => {
        if (cancelled) return;
        const parsed = parseEnsemble(doc);
        $ensemble.value = parsed;
        $error.value = null;
        // the field's fan is drawn over the document's OWN members, never
        // over whatever runs happen to be on screen
        appStore.getState().setEnsemble(parsed.ensembleId, parsed.runIds);
      })
      .catch((err) => {
        if (cancelled) return;
        // an absent ensemble reaches the panel as an error, never as an empty
        // fan: on screen those look identical and only one is a measurement
        $ensemble.value = null;
        $error.value = err instanceof Error ? err.message : String(err);
      });
    return () => {
      cancelled = true;
    };
  }, [ensembleId]);

  if ($ids.value !== null && $ids.value.length === 0 && !$ensemble.value && !$error.value) {
    return null; // no ensembles in this store; the panel is optional furniture
  }
  if (!ensembleId && !$error.value) return null;

  const e = $ensemble.value;
  const disagreements = e ? intervalDisagreement(e) : [];
  const missing = e ? missingLines(e) : [];
  const short = e ? shortfall(e) : 0;

  return (
    <section class={`ens-panel${$collapsed.value ? " ens-collapsed" : ""}`}>
      <header class="ens-head">
        <h3 class="ens-title">The fan — N runs of one protocol</h3>
        <button
          class="ens-toggle"
          onClick={() => ($collapsed.value = !$collapsed.value)}
          aria-expanded={!$collapsed.value}
        >
          {$collapsed.value ? "show" : "hide"}
        </button>
      </header>

      {$collapsed.value ? null : $error.value ? (
        <p class="ens-note ens-note-missing">{$error.value}</p>
      ) : !e ? (
        <p class="ens-note ens-dim">reading the ensemble…</p>
      ) : (
        <>
          <p class="ens-provenance ens-mono">
            {e.ensembleId} · {e.nRuns} run{e.nRuns === 1 ? "" : "s"}
            {short > 0 ? ` of ${e.nRunsRequested} requested` : ""} ·{" "}
            {e.conditions.map((c) => `${c.condition} ${c.nRuns}`).join(" / ")}
            {e.protocolId ? ` · ${e.protocolId}` : ""}
          </p>
          {short > 0 ? (
            <p class="ens-note ens-note-warn">
              {short} run{short === 1 ? "" : "s"} named in the manifest {short === 1 ? "is" : "are"}{" "}
              not in the store and {short === 1 ? "is" : "are"} excluded from every statistic
              here. n is what was measured, not what was asked for.
            </p>
          ) : null}
          <p class="ens-note ens-dim">
            Seed base {e.seedBase ?? "—"},{" "}
            {e.seedApplied
              ? "applied to the runs"
              : "recorded and NOT applied — no agent here accepts a seed; it seeds the split-half draws only"}
            .
          </p>

          <Fan e={e} />

          <section class="ens-block">
            <h4 class="ens-h">Rates across the {e.nRuns} runs</h4>
            <p class="ens-note ens-dim">
              A point estimate needs {Number.isFinite(e.pointEstimateMinRuns) ? e.pointEstimateMinRuns : "a threshold this document does not state"} runs.
              Anything below that is shown as an interval and its raw counts.
            </p>
            {e.rates.map((r) => (
              <RateRow key={r.key} e={e} rate={r} />
            ))}
          </section>

          <Reliability e={e} />

          {missing.length ? (
            <section class="ens-block">
              <h4 class="ens-h">Not computed</h4>
              <ul class="ens-missing-list">
                {missing.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            </section>
          ) : null}

          {disagreements.length ? (
            <section class="ens-block ens-block-alarm">
              <h4 class="ens-h">The file and the formula disagree</h4>
              <ul class="ens-missing-list">
                {disagreements.map((d) => (
                  <li key={d}>{d}</li>
                ))}
              </ul>
            </section>
          ) : null}
        </>
      )}
    </section>
  );
}
