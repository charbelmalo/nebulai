/** SteerRail — the text half of the intervention figure (#26).
 *
 *  The stage next to it draws how FAR the model's next-token distribution
 *  moved. This panel says WHAT moved: one slider over the measured strengths,
 *  and the two generations side by side — the un-hooked baseline on the left,
 *  the hooked run on the right, diverging from a common prefix that is drawn
 *  as a common prefix.
 *
 *  Four rules this panel keeps.
 *
 *  · **α = 0 is a position on the slider, not a missing one.** At that stop the
 *    right-hand column is the control: no hook was installed, and the bundle
 *    carries a per-row assertion that its logits were bit-identical to the
 *    baseline. The panel prints that assertion rather than implying it by
 *    showing two identical strings.
 *  · **Both columns come from the same file.** Nothing here re-runs the model
 *    or interpolates between the alphas that were run; moving the slider
 *    selects a measured cell, and the stage's probe moves to the same cell.
 *  · **The target scores are shown even when they embarrass the figure.** On
 *    the shipped Golden Gate sweep the teacher-forced score for the completion
 *    the intervention was aimed at goes DOWN, and further down than the control
 *    completion does. That is the finding; a rail that hid it would be
 *    advertising.
 *  · **The claim sentence is the producer's, verbatim.** It is generated from
 *    the measured numbers in the intervention's own terms (§2.4 / D3) — never
 *    restated here in stronger words, and never turned into a sentence about
 *    what the feature *is*.
 *
 *  **Travel along a phase-1 axis, and its two separate refusals.** A
 *  direction-verb bundle (`add` / `ablate`) now carries, per α, where the
 *  prompts sat ON the direction before and after the hook — `intervene.py`'s
 *  `axis` block. The rail prints that pair and offers the link that engages the
 *  same direction on the map. Both halves can fail, for different reasons, and
 *  the panel keeps them apart:
 *
 *  · the bundle may have no axis at all — `clamp` pins an SAE feature and `cap`
 *    clips a box, and neither names a direction to project onto. The shipped
 *    Golden Gate sweep is a `clamp`, so on it this section shows the producer's
 *    refusal sentence and no numbers.
 *  · the MAP may be unable to lay out along the direction even when the
 *    intervention measured it perfectly. `refusal-style-v1-L8` lives in
 *    `resid.L8`; the gpt2 map's points are `W_E.centered` token embeddings, so
 *    D2 refuses the projection and `data/directions.ts` says so in the same
 *    words the CLI uses. The numbers still show; the link does not.
 *
 *  A link that looked live and did nothing would be worse than no link, and a
 *  single "axis unavailable" would hide which of the two walls was hit.
 */

import { useEffect } from "preact/hooks";
import { useSignal } from "@preact/signals";
import {
  commonPrefix,
  fmtAlpha,
  onSteer,
  selectSteer,
  steerAxis,
  type SteerCell,
} from "../scene/interp/steer";
import type { InterveneBundle, InterveneRow, InterveneRun } from "../data/interp";
import { appStore } from "../app/store";
import { $channels } from "../data/channels";
import { $directions } from "../data/directions";
import { $datasetId } from "./state";

/** The axis section: measured travel, then either the map link or the reason.
 *
 *  `t = 1` rather than a blend because the point of following the link is to see
 *  the layout the axis implies; the rail's own slider is then there to come back.
 */
function AxisSection(props: { bundle: InterveneBundle; sel: SteerCell }) {
  // both loader signals, because the map half of the gate reads directions AND
  // channels — a section watching only one would sit on a stale refusal
  void $directions.value;
  void $channels.value;
  const dsId = $datasetId.value;
  const ax = steerAxis(props.bundle, dsId, props.sel);
  if (!ax.ready) {
    return (
      <p class="steer-axis-refused is-section" data-testid="steer-axis-refused">
        {ax.reason}
      </p>
    );
  }
  const st = appStore.getState();
  const here = ax.here;
  return (
    <div class="steer-axis" data-testid="steer-axis">
      <div class="steer-axis-head">
        <span class="steer-axis-k">on the axis</span>
        <span class="steer-axis-id">
          {ax.directionId} · {ax.space} · hook L{ax.layer}
        </span>
      </div>
      {here ? (
        <p class="steer-axis-travel">
          {here.projBaseline.toFixed(2)} → {here.projIntervened.toFixed(2)} (
          {here.projDelta >= 0 ? "+" : ""}
          {here.projDelta.toFixed(2)} along) · perpendicular{" "}
          {here.orthDelta >= 0 ? "+" : ""}
          {here.orthDelta.toFixed(2)}
        </p>
      ) : (
        <p class="steer-axis-travel">this α has no axis row</p>
      )}
      {ax.mapLink ? (
        <button
          type="button"
          class="steer-axis-link"
          onClick={() => {
            st.setAxisDirection(ax.mapLink!.directionId);
            st.setAxisT(1);
          }}
        >
          lay the map out along this axis
        </button>
      ) : (
        <p class="steer-axis-refused" data-testid="steer-axis-map-refused">
          {ax.mapRefusal}
        </p>
      )}
    </div>
  );
}

/** A token string as it should read in prose. GPT-2's byte-BPE writes a
 *  leading space into the token itself, and a column of `Ġ`-style artefacts
 *  would be the tokenizer's business leaking into the reader's. */
function joinTokens(toks: string[], from: number, to: number): string {
  return toks.slice(from, to).join("");
}

function Generation(props: {
  title: string;
  sub: string;
  prompt: string;
  toks: string[];
  shared: number;
  tone: "base" | "steer" | "control";
}) {
  const { toks, shared } = props;
  return (
    <div class={`steer-gen is-${props.tone}`}>
      <div class="steer-gen-head">
        <span class="steer-gen-title">{props.title}</span>
        <span class="steer-gen-sub">{props.sub}</span>
      </div>
      <p class="steer-gen-text">
        <span class="steer-gen-prompt">{props.prompt}</span>
        {shared > 0 && <span class="steer-gen-same">{joinTokens(toks, 0, shared)}</span>}
        <span class="steer-gen-diff">{joinTokens(toks, shared, toks.length)}</span>
      </p>
    </div>
  );
}

/** The identity line: exactly what was done, in the producer's protocol string,
 *  at the strength currently selected. */
function protocolOf(row: InterveneRow | undefined): string {
  return row?.protocol ?? "";
}

export function SteerRail() {
  const bundle = useSignal<InterveneBundle | null>(null);
  const cell = useSignal<SteerCell | null>(null);

  useEffect(
    () =>
      onSteer((b, sel) => {
        bundle.value = b;
        cell.value = sel;
      }),
    [],
  );

  const b = bundle.value;
  const sel = cell.value;
  if (!b || !sel) return null;
  const row: InterveneRow | undefined = b.rows[sel.row];
  const run: InterveneRun | undefined = row?.runs[sel.col];
  if (!row || !run) return null;

  const shared = commonPrefix(run.baseline.tokens, run.intervened.tokens);
  const isControl = row.is_identity;
  const broken = isControl && !row.identical_to_baseline;
  const dir = b.meta?.direction;

  return (
    <section class="steer-rail" aria-label="Intervention readout">
      <header class="steer-head">
        <h2 class="steer-title">
          {b.verb} · α {fmtAlpha(row.alpha)}
        </h2>
        <span class="steer-proto" title={protocolOf(row)}>
          {protocolOf(row)}
        </span>
      </header>

      {/* Prompt picker. The sweep ran every prompt at every strength, so this
          is a selection among measurements, never a new run. */}
      <div class="steer-prompts" role="radiogroup" aria-label="Prompt">
        {b.prompts.map((p, i) => (
          <button
            key={p}
            type="button"
            role="radio"
            aria-checked={i === sel.col}
            class={`steer-prompt${i === sel.col ? " is-active" : ""}`}
            title={p}
            onClick={() => selectSteer({ row: sel.row, col: i })}
          >
            {p}
          </button>
        ))}
      </div>

      {/* One slider, over the measured strengths only: `step` is an index, so
          the handle cannot come to rest between two runs that exist. */}
      <label class="steer-slider">
        <span class="steer-slider-k">off</span>
        <input
          type="range"
          min="0"
          max={String(Math.max(0, b.rows.length - 1))}
          step="1"
          value={String(sel.row)}
          aria-label="Intervention strength"
          aria-valuetext={`alpha ${fmtAlpha(row.alpha)}${isControl ? ", control" : ""}`}
          onInput={(e) =>
            selectSteer({
              row: Number((e.currentTarget as HTMLInputElement).value),
              col: sel.col,
            })
          }
        />
        <span class="steer-slider-k">α {fmtAlpha(b.rows[b.rows.length - 1]?.alpha ?? 1)}</span>
      </label>
      <div class="steer-ticks" aria-hidden="true">
        {b.rows.map((r, i) => (
          <span
            key={r.alpha}
            class={`steer-tick${i === sel.row ? " is-active" : ""}${r.is_identity ? " is-ctrl" : ""}`}
          >
            {fmtAlpha(r.alpha)}
          </span>
        ))}
      </div>

      <div class="steer-gens">
        <Generation
          title="baseline"
          sub="no hook"
          prompt={run.prompt}
          toks={run.baseline.tokens}
          shared={shared}
          tone="base"
        />
        <Generation
          title={isControl ? "control" : "intervened"}
          sub={
            isControl
              ? broken
                ? "α = 0 but NOT identical — broken harness"
                : "α = 0 · no hook installed · logits bit-identical"
              : `${run.kl_bits.toFixed(3)} bits of KL`
          }
          prompt={run.prompt}
          toks={run.intervened.tokens}
          shared={shared}
          tone={isControl ? "control" : "steer"}
        />
      </div>

      {/* The numbers the argmax hides. Kept whichever way they point. */}
      {run.targets && run.targets.length > 0 && (
        <table class="steer-targets">
          <caption>
            teacher-forced log-probability of a fixed completion — before → after
          </caption>
          <tbody>
            {run.targets.map((t) => {
              const d = t.intervened_logprob - t.baseline_logprob;
              return (
                <tr key={t.text}>
                  <th scope="row">{t.text.trim()}</th>
                  <td>{t.baseline_logprob.toFixed(2)}</td>
                  <td>{t.intervened_logprob.toFixed(2)}</td>
                  <td class={d >= 0 ? "is-up" : "is-down"}>
                    {d >= 0 ? "+" : ""}
                    {d.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      <dl class="steer-facts">
        <div>
          <dt>‖resid‖ at the final position</dt>
          <dd>
            {run.resid_norm_baseline.toFixed(1)} → {run.resid_norm_intervened.toFixed(1)}
          </dd>
        </div>
        <div>
          <dt>decoding</dt>
          <dd>{b.notes?.decoding ?? run.intervened.decoding}</dd>
        </div>
        {dir && (
          <div>
            <dt>direction</dt>
            <dd>
              {dir.label} · {dir.space} · {dir.method}
            </dd>
          </div>
        )}
      </dl>

      <AxisSection bundle={b} sel={sel} />

      <p class={`steer-claim${broken ? " is-broken" : ""}`}>
        {broken
          ? "This sweep's α = 0 row is not bit-identical to the baseline, so it " +
            "has no control and no claim. Read no effect size from it."
          : b.claim}
      </p>
      <p class="steer-d6">{b.notes?.d6 ?? "inference-time hooks only; no weights were written"}</p>
    </section>
  );
}
