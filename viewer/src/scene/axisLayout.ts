/** Where a point goes when the map is laid out on a direction.
 *
 *  The axis layout is a second set of 2-D coordinates for the same points:
 *  **x = the component along the direction, y = the component orthogonal to
 *  it**. Nothing is re-embedded, nothing is re-clustered, no optimiser runs —
 *  it is the same 768-dimensional point, resolved onto one named vector and
 *  the subspace that vector leaves behind. That is why the atlas blends into
 *  it rather than switching to it (R1: a layout state is not a new coordinate
 *  system; `viewMode` stays `"atlas"` throughout).
 *
 *  Two properties this module exists to guarantee, both of which a
 *  "just normalise in the shader" version would quietly break:
 *
 *  · **The real layout and the null layout share one ruler.** The extents are
 *    taken over the union of both, so a ghost that sits inside the real cloud
 *    is genuinely inside it and not merely drawn at the same size. A separate
 *    per-cloud normalisation would make every null look exactly as spread as
 *    its direction, which is the one thing the ghost exists to disprove.
 *  · **A point with no measured projection does not move.** It stays at its map
 *    position and is counted. Sliding it to 0 would place it at the middle of
 *    the axis — a confident claim about a point nobody measured.
 */

/** The map's own footprint, so the axis layout lands in the same frame the
 *  camera is already looking at. */
export interface MapBounds {
  cx: number;
  cy: number;
  halfW: number;
  halfH: number;
}

/** Bounding box of the `pos2` layout, as centre + half-extents. Degenerate
 *  axes (a map one point wide) get a half-extent of 1 rather than 0, so the
 *  blend never divides by nothing. */
export function mapBounds(pos2: Float32Array): MapBounds {
  let xlo = Number.POSITIVE_INFINITY;
  let xhi = Number.NEGATIVE_INFINITY;
  let ylo = Number.POSITIVE_INFINITY;
  let yhi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i + 1 < pos2.length; i += 2) {
    const x = pos2[i]!;
    const y = pos2[i + 1]!;
    if (Number.isFinite(x)) {
      if (x < xlo) xlo = x;
      if (x > xhi) xhi = x;
    }
    if (Number.isFinite(y)) {
      if (y < ylo) ylo = y;
      if (y > yhi) yhi = y;
    }
  }
  if (!Number.isFinite(xlo) || !Number.isFinite(ylo)) {
    return { cx: 0, cy: 0, halfW: 1, halfH: 1 };
  }
  return {
    cx: (xlo + xhi) / 2,
    cy: (ylo + yhi) / 2,
    halfW: Math.max((xhi - xlo) / 2, 1e-6),
    halfH: Math.max((yhi - ylo) / 2, 1e-6),
  };
}

/** The shared [lo, hi] of however many columns are handed in, over measured
 *  values only. Returns null when nothing at all was measured. */
export function sharedRange(...cols: (Float32Array | null | undefined)[]): [number, number] | null {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const c of cols) {
    if (!c) continue;
    for (let i = 0; i < c.length; i++) {
      const v = c[i]!;
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  if (hi <= lo) return [lo - 0.5, lo + 0.5];
  return [lo, hi];
}

export interface AxisLayout {
  /** n × 4: (realX, realY, nullX, nullY) in world units, ready for `iAxis` */
  positions: Float32Array;
  /** how many points had no measured projection and were left at their map
   *  position. Reported, never hidden — and never rendered as a position. */
  nMissing: number;
  /** the shared rulers, for the rail's readout */
  parRange: [number, number];
  orthRange: [number, number];
}

/** Build the packed `iAxis` buffer.
 *
 *  `pos2` is the map layout the unmeasured points fall back to; the four
 *  columns are the direction's parallel/orthogonal channels and its null's. All
 *  four are required: an axis drawn without its ghost is the figure R5 forbids,
 *  so this function has no single-cloud mode to reach for.
 */
export function axisLayout(
  pos2: Float32Array,
  par: Float32Array,
  orth: Float32Array,
  nullPar: Float32Array,
  nullOrth: Float32Array,
  bounds: MapBounds,
): AxisLayout {
  const n = Math.floor(pos2.length / 2);
  const positions = new Float32Array(n * 4);
  const pr = sharedRange(par, nullPar) ?? [0, 1];
  const or = sharedRange(orth, nullOrth) ?? [0, 1];
  const pSpan = pr[1] - pr[0];
  const oSpan = or[1] - or[0];

  let nMissing = 0;
  for (let i = 0; i < n; i++) {
    const bx = pos2[i * 2]!;
    const by = pos2[i * 2 + 1]!;
    const p = par[i];
    const o = orth[i];
    const np = nullPar[i];
    const no = nullOrth[i];
    const realOk = p !== undefined && o !== undefined && Number.isFinite(p) && Number.isFinite(o);
    const nullOk =
      np !== undefined && no !== undefined && Number.isFinite(np) && Number.isFinite(no);
    if (!realOk || !nullOk) nMissing++;
    // x spans the map's width, y its height, so the blend is a rearrangement
    // inside the frame the camera already holds rather than a flight somewhere
    positions[i * 4] = realOk
      ? bounds.cx + (((p! - pr[0]) / pSpan) * 2 - 1) * bounds.halfW
      : bx;
    positions[i * 4 + 1] = realOk
      ? bounds.cy + (((o! - or[0]) / oSpan) * 2 - 1) * bounds.halfH
      : by;
    positions[i * 4 + 2] = nullOk
      ? bounds.cx + (((np! - pr[0]) / pSpan) * 2 - 1) * bounds.halfW
      : bx;
    positions[i * 4 + 3] = nullOk
      ? bounds.cy + (((no! - or[0]) / oSpan) * 2 - 1) * bounds.halfH
      : by;
  }
  return { positions, nMissing, parRange: pr, orthRange: or };
}

/** Where point `i` actually is at blend `t`, on the CPU.
 *
 *  The hover readout and the tooltip must agree with the shader to the pixel,
 *  so this is the same arithmetic the vertex node does — `mix(base, axis, t)`,
 *  where `base` is itself `mix(pos2, pos3, morph)`. Keeping it in this module
 *  (rather than inline in the driver) is what lets a unit test pin the two
 *  together without a GPU.
 */
export function blendedPosition(
  i: number,
  pos2: Float32Array,
  pos3: Float32Array,
  morph: number,
  axis: Float32Array | null,
  t: number,
  lane: "real" | "null" = "real",
): [number, number, number] {
  const m = Math.min(1, Math.max(0, morph));
  const bx = pos2[i * 2]! * (1 - m) + pos3[i * 3]! * m;
  const by = pos2[i * 2 + 1]! * (1 - m) + pos3[i * 3 + 1]! * m;
  const bz = pos3[i * 3 + 2]! * m;
  const a = Math.min(1, Math.max(0, t));
  if (!axis || a <= 0) return [bx, by, bz];
  const o = lane === "real" ? 0 : 2;
  const ax = axis[i * 4 + o];
  const ay = axis[i * 4 + o + 1];
  if (ax === undefined || ay === undefined) return [bx, by, bz];
  // the axis layout is flat: z eases to 0 with everything else, so an axis at
  // t = 1 reads the same whether you arrived from the 2-D map or the flythrough
  return [bx * (1 - a) + ax * a, by * (1 - a) + ay * a, bz * (1 - a)];
}
