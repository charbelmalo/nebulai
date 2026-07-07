/** Per-cluster hulls + label anchors, computed in the worker after
 *  columnarization. Convex (Andrew monotone chain) for now — deterministic
 *  and cheap; the territories layer can swell these visually. */

export interface ClusterHull {
  clusterId: number;
  /** hull vertices, flat [x0,y0, x1,y1, ...] in pos2 space, CCW */
  ring: Float32Array;
  /** label anchor = member centroid in pos2 space (not hull centroid — less
   *  swayed by outlier vertices) */
  anchor: [number, number];
  size: number;
}

export function computeHulls(
  pos2: Float32Array,
  clusterId: Int32Array,
): ClusterHull[] {
  const members = new Map<number, number[]>();
  for (let i = 0; i < clusterId.length; i++) {
    const c = clusterId[i]!;
    if (c < 0) continue; // noise has no territory
    let arr = members.get(c);
    if (!arr) members.set(c, (arr = []));
    arr.push(i);
  }

  const hulls: ClusterHull[] = [];
  for (const [cid, idx] of [...members.entries()].sort((a, b) => a[0] - b[0])) {
    let sx = 0;
    let sy = 0;
    const pts: [number, number][] = new Array(idx.length);
    for (let j = 0; j < idx.length; j++) {
      const x = pos2[idx[j]! * 2]!;
      const y = pos2[idx[j]! * 2 + 1]!;
      pts[j] = [x, y];
      sx += x;
      sy += y;
    }
    const ringPts = convexHull(pts);
    const ring = new Float32Array(ringPts.length * 2);
    for (let j = 0; j < ringPts.length; j++) {
      ring[j * 2] = ringPts[j]![0];
      ring[j * 2 + 1] = ringPts[j]![1];
    }
    hulls.push({
      clusterId: cid,
      ring,
      anchor: [sx / idx.length, sy / idx.length],
      size: idx.length,
    });
  }
  return hulls;
}

/** Max distance from the label anchor to any hull vertex, in pos2 units —
 *  the projected-size proxy used for zoom-band label culling and fly-to. */
export function hullRadius(h: ClusterHull): number {
  let r = 0;
  for (let j = 0; j < h.ring.length / 2; j++) {
    const dx = h.ring[j * 2]! - h.anchor[0];
    const dy = h.ring[j * 2 + 1]! - h.anchor[1];
    r = Math.max(r, Math.hypot(dx, dy));
  }
  return r;
}

/** Andrew's monotone chain, CCW output. Handles n < 3 by returning the input. */
export function convexHull(pts: [number, number][]): [number, number][] {
  if (pts.length <= 2) return [...pts];
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const lower: [number, number][] = [];
  for (const pt of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, pt) <= 0)
      lower.pop();
    lower.push(pt);
  }
  const upper: [number, number][] = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const pt = p[i]!;
    while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, pt) <= 0)
      upper.pop();
    upper.push(pt);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}
