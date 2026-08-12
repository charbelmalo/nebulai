/** StatStrip — the labelled-scalar tiles used under a chart and inside the
 *  Sessions cards. Lifted out of SessionsPage, which had the only copy, and
 *  now lives in src/viz alongside ChartCard, its main consumer, so the interp
 *  drivers can reach it too without depending on chrome/.
 *
 *  Two exports, because there are two real shapes:
 *    · `<Stat>`      one tile. Drop into any grid — Sessions packs them 3-up
 *                    inside `.stat-row`.
 *    · `<StatStrip>` the footer strip that spans a chart's full width, tiles
 *                    separated by hairlines.
 *
 *  Anatomy is fixed and deliberate: LARGE tabular number, tiny uppercase label
 *  under it. Tabular figures matter — these sit in a row and a proportional
 *  font makes the column edges shimmer as values change during playback.
 *
 *  HONESTY: a tile shows a value the view actually computed. `value` is
 *  pre-formatted by the caller because the caller owns the units and the
 *  precision — this component never rounds, scales, or derives anything. If a
 *  quantity is unavailable, pass "—" rather than a zero, which would read as a
 *  measurement. */

export interface StatTile {
  /** Tiny uppercase caption under the number. */
  label: string;
  /** Pre-formatted display value — caller owns units and precision. "—" when
   *  genuinely unavailable (never a stand-in zero). */
  value: string;
  /** Optional hover text: the full-precision value, or where it came from. */
  title?: string;
}

export function Stat(props: StatTile) {
  return (
    <div class="stat" title={props.title}>
      <span class="stat-v">{props.value}</span>
      <span class="stat-l">{props.label}</span>
    </div>
  );
}

/** Full-width footer strip under a chart. `label` names the strip for screen
 *  readers (e.g. "Attention rollout summary"); the tiles themselves are read
 *  as value-then-label pairs in DOM order. */
export function StatStrip(props: { tiles: StatTile[]; label?: string }) {
  if (props.tiles.length === 0) return null;
  return (
    <div class="stat-strip" role="group" aria-label={props.label ?? "Summary statistics"}>
      {props.tiles.map((t) => (
        <Stat key={t.label} label={t.label} value={t.value} title={t.title} />
      ))}
    </div>
  );
}
