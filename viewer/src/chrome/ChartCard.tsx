/** ChartCard — the per-chart frame: a real header ABOVE the canvas, the canvas,
 *  and the footer stat strip below it.
 *
 *  Why a header at all, when the Internals page already had a legend card: the
 *  legend floats OVER the plot, so a chart's identity, its provenance, and its
 *  controls were all occluding the very data they describe, and the user had to
 *  collapse them to see the thing. Splitting that in two along the line of what
 *  each part is FOR:
 *
 *    · header (here, in flow)  — which chart this is, what it plots, its
 *                                controls. Never covers data, always visible.
 *    · legend  (still floating) — the ENCODING key: colour swatches, the units,
 *                                the honesty note. That has to sit next to the
 *                                marks it decodes, so it stays on the canvas.
 *
 *  The header is deliberately not collapsible. A chart that can hide what
 *  quantity it is showing is a chart that can be screenshotted without it.
 *
 *  Pure chrome — this component knows nothing about any driver. */

import type { ComponentChildren } from "preact";
import { StatStrip, type StatTile } from "./StatStrip";

export function ChartCard(props: {
  /** Feature number in the 25-feature spec, shown as a lead-in chip. */
  n?: number;
  title: string;
  /** One plain-language line: what real quantity is on screen. Not marketing —
   *  it must be supported by the feature's own blurb/math/source. */
  subtitle?: string;
  /** Small muted tag after the title (the model id, or the feature group). */
  tag?: string;
  /** Controls docked top-right: pickers, download, toggles. */
  controls?: ComponentChildren;
  /** The canvas host and anything absolutely positioned over it. Children are
   *  wrapped in `.chart-card-body`, which is the positioning context for those
   *  overlays — see the note on the render below. */
  children: ComponentChildren;
  /** Footer tiles. Omitted or empty renders no strip at all — a chart with
   *  nothing honest to summarise gets no summary, rather than a row of zeros. */
  tiles?: StatTile[];
  /** Extra class on the root, for page-specific layout. */
  class?: string;
}) {
  const tiles = props.tiles ?? [];
  return (
    <div class={`chart-card${props.class ? ` ${props.class}` : ""}`}>
      <header class="chart-card-head">
        <div class="chart-card-id">
          <h3 class="chart-card-title">
            {props.n != null && <span class="chart-card-n">#{props.n}</span>}
            {props.title}
            {props.tag && <span class="chart-card-tag">{props.tag}</span>}
          </h3>
          {props.subtitle && <p class="chart-card-sub">{props.subtitle}</p>}
        </div>
        {props.controls && <div class="chart-card-controls">{props.controls}</div>}
      </header>
      {/* The body is `position: relative` and is the ONLY positioning context
          a caller's floating overlays should see: its box is exactly the plot
          area — header above it, stat strip below it. So a legend docked
          `bottom: 24px` clears the strip and a spinner at `top: 50%` centres
          on the data, both without knowing either chrome height. Anchoring to
          the card (or the page's stage) instead would make every overlay carry
          a hardcoded offset that goes stale the moment a subtitle wraps. */}
      <div class="chart-card-body">{props.children}</div>
      <StatStrip tiles={tiles} label={`${props.title} — summary`} />
    </div>
  );
}
