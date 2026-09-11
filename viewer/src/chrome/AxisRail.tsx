/** AxisRail — the direction axis: pick one, blend the map onto it, and never
 *  see it without its null.
 *
 *  The rail is built around one refusal and one picture.
 *
 *  The refusal is `data/directions.ts`'s: a direction with no null, or whose
 *  projection channel is missing or in another space, is not offered at all —
 *  it is listed under "not renderable" with the reason in the same words the
 *  CLI uses. There is no "show it anyway".
 *
 *  The picture is two histograms on ONE ruler: the real projection solid, the
 *  null's ghosted behind it. That is R5 rendered rather than asserted — the
 *  ghost is in the frame of every screenshot, so a figure cannot be cropped
 *  into a claim. Both come from `histogram()` with a shared range, because two
 *  histograms auto-scaled to their own extents would make every null look
 *  exactly as spread as its direction.
 *
 *  The stat strip carries the two numbers that mean different things, and the
 *  copy keeps them apart: the **held-out** separation of the direction's own
 *  two sets (the claim it makes) and the **map-wide** overlap with its null
 *  (usually large, and that is a finding, not a failure). No sentence here
 *  says the model uses this axis for anything — D3's one causal sentence is
 *  spent on the intervention figures, not on a projection.
 */

import { appStore } from "../app/store";
import { $channels, channelsFor } from "../data/channels";
import {
  $directions,
  axisClaim,
  axisMapClaim,
  axisChannels,
  histogram,
  histogramRange,
  renderableDirections,
  type Direction,
} from "../data/directions";
import { ChartCard } from "@psychix/viz/ChartCard";
import type { StatTile } from "@psychix/viz/StatStrip";
import { $axis, $datasetId } from "./state";

const BINS = 44;
const PLOT_W = 260;
const PLOT_H = 72;

/** Pre-formatted, or the em dash. Never 0 for "not measured" (§2.2). */
function fmt(v: number | null, digits = 2): string {
  return v === null ? "—" : v.toFixed(digits);
}

export function AxisRail() {
  // Touching both loader signals is what re-renders this panel when either
  // sidecar lands after boot. The gate reads BOTH — directions for the
  // vectors, channels for the projections they were supposed to produce — so
  // a component that watched only one would sit on a stale refusal.
  void $directions.value;
  void $channels.value;
  const dsId = $datasetId.value;
  const ui = $axis.value;
  const { ok, drops } = renderableDirections(dsId);
  if (ok.length === 0 && drops.length === 0) return null;

  const st = appStore.getState();
  const active = ui.directionId ? ok.find((d) => d.id === ui.directionId) ?? null : null;

  return (
    <section class="axis-rail" aria-label="Direction axis">
      <header class="axis-head">
        <h2 class="axis-title">Directions</h2>
        <span class="axis-count">
          {ok.length} of {ok.length + drops.length} renderable
        </span>
      </header>

      <div class="lens-chips">
        {ok.map((d) => {
          const on = ui.directionId === d.id;
          return (
            <button
              type="button"
              key={d.id}
              class={on ? "lens-chip is-on" : "lens-chip"}
              aria-pressed={on}
              title={`${d.id} · ${d.method} · space ${d.space} · ${d.source.protocol}`}
              onClick={() => st.setAxisDirection(on ? null : d.id)}
            >
              {d.label}
            </button>
          );
        })}
      </div>

      {active && (
        <>
          <AxisFigure dir={active} />
          <AxisControls />
        </>
      )}

      {drops.length > 0 && (
        <ul class="axis-drops">
          {drops.map((x) => (
            <li class="axis-drop" key={x.direction.id}>
              <span class="axis-drop-id">{x.direction.id}</span>
              <span class="axis-drop-why">{x.reason}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** The blend control, the two histograms, and the numbers under them. */
function AxisFigure({ dir }: { dir: Direction }) {
  const dsId = $datasetId.value;
  const found = axisChannels(dsId, dir.id);
  if (!found) return null;

  const [lo, hi] = histogramRange(found.par, found.nullPar);
  const real = histogram(found.par, lo, hi, BINS);
  const nul = histogram(found.nullPar, lo, hi, BINS);
  const peak = Math.max(...real.bins, ...nul.bins, 1e-9);

  const c = dir.source.contrast;
  const s = dir.projection?.stats;
  const tiles: StatTile[] = [
    {
      label: "held-out d",
      value: c && c.heldout_cohens_d !== null ? fmt(c.heldout_cohens_d) : "—",
      title:
        c && c.heldout_cohens_d !== null
          ? `Cohen's d between the direction's own two sets, refitted on half of each and scored on the other half (n = ${c.heldout_n_pos ?? 0}/${c.heldout_n_neg ?? 0})`
          : "not measured — a set of fewer than four members cannot be split",
    },
    {
      label: "random |d|",
      value: c ? fmt(c.null_cohens_d_mean) : "—",
      title: c
        ? `mean |d| of ${c.null_n ?? 0} random unit directions on the same two sets — the control for the number on its left`
        : "no random-direction baseline was recorded",
    },
    {
      label: "map overlap",
      value: s ? fmt(s.overlap, 3) : "—",
      title:
        "histogram intersection of the real projection and its null across every point in the map. 1.000 = the same picture.",
    },
    { label: "n", value: s && s.n !== null ? s.n.toLocaleString() : "—" },
  ];

  const path = (bins: Float64Array) => {
    const w = PLOT_W / bins.length;
    let d = `M 0 ${PLOT_H}`;
    for (let i = 0; i < bins.length; i++) {
      const h = (bins[i]! / peak) * (PLOT_H - 2);
      d += ` L ${(i * w).toFixed(2)} ${(PLOT_H - h).toFixed(2)} L ${((i + 1) * w).toFixed(2)} ${(PLOT_H - h).toFixed(2)}`;
    }
    return `${d} L ${PLOT_W} ${PLOT_H} Z`;
  };

  return (
    <ChartCard
      title={dir.label}
      tag={dir.space}
      subtitle={`⟨point, direction⟩ across the map — real solid, ${dir.null?.n ?? 0} random unit directions ghosted behind it`}
      tiles={tiles}
      class="axis-card"
    >
      <svg
        class="axis-hist"
        viewBox={`0 0 ${PLOT_W} ${PLOT_H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`Projection histogram for ${dir.label}, with its null`}
      >
        <path class="axis-hist-null" d={path(nul.bins)} />
        <path class="axis-hist-real" d={path(real.bins)} />
      </svg>
      <div class="axis-range">
        <span>{lo.toFixed(2)}</span>
        <span>{hi.toFixed(2)}</span>
      </div>
    </ChartCard>
  );
}

/** The blend slider and the null switch, kept OUT of the card so the card is
 *  exactly the figure and nothing else — a screenshot of it carries the ghost
 *  and the numbers and no controls. */
export function AxisControls() {
  void $directions.value;
  void $channels.value;
  const dsId = $datasetId.value;
  const ui = $axis.value;
  const st = appStore.getState();
  const dir = ui.directionId ? renderableDirections(dsId).ok.find((d) => d.id === ui.directionId) : null;
  if (!dir) return null;
  return (
    <div class="axis-controls">
      <label class="axis-slider">
        <span class="axis-slider-k">map</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={String(ui.t)}
          aria-label="Blend the map onto this direction"
          onInput={(e) => st.setAxisT(Number((e.currentTarget as HTMLInputElement).value))}
        />
        <span class="axis-slider-k">axis</span>
      </label>
      <label class="axis-ghost-toggle">
        <input
          type="checkbox"
          checked={ui.showNull}
          onChange={(e) => st.setAxisNull((e.currentTarget as HTMLInputElement).checked)}
        />
        <span>show the null cloud</span>
      </label>
      <p class="axis-claim">{axisClaim(dir)}</p>
      <p class="axis-claim axis-claim-map">{axisMapClaim(dir)}</p>
      {channelsFor(dsId) === null && (
        <p class="axis-claim">this map's channels have not been read, so nothing is drawn</p>
      )}
    </div>
  );
}
