/** Columnarize a parsed nebulai.json into typed arrays for zero-copy transfer
 *  out of the parse worker and direct GPU upload. Pure function — unit-tested
 *  in node without a DOM. */

import type { NebulaiCluster, NebulaiDoc, NebulaiMeta } from "./schema";
import { schemaVersion } from "./schema";

export interface EdgeColumns {
  space: string;
  metric: string;
  sigma: number;
  /** flat triples [a, b, weight] × nClusterEdges */
  clusterEdges: Float32Array;
  /** per-point kNN, flat n*k; null when exported with cluster-only edges */
  knn: { k: number; sigma: number; ids: Int32Array; sims: Float32Array } | null;
}

/** Exact source identity and evidence, kept on the CPU beside the render
 *  buffers. The Float32/Uint8 arrays above are for drawing; they cannot
 *  regenerate a source record (Float32 rounds coordinates, the Uint8 opacity
 *  quantizes membership). These Float64 columns hold every source number as
 *  JSON.parse produced it, so a finding's evidence round-trips exactly.
 *
 *  Rows stay the renderer's index space — picking, search and hulls all keep
 *  using row numbers. A row is resolved to its source identity here, and only
 *  here, when something needs to name the unit (inspector, export, replay). */
export interface SourceColumns {
  /** source `point.id`; NaN where the export has none */
  pointId: Float64Array;
  /** index into `unitKinds`; -1 where the export has no unit_ref */
  unitKindIdx: Int32Array;
  /** interned raw `unit_ref.kind` strings */
  unitKinds: string[];
  /** source `unit_ref.index`; NaN where missing */
  unitIndex: Float64Array;
  /** source layer; NaN = null (token maps have no layer) */
  layer: Float64Array;
  /** exact source coordinates, n*2 and n*3 */
  xy: Float64Array;
  xyz: Float64Array;
  /** exact HDBSCAN membership (`confidence` in the file) */
  membership: Float64Array;
  /** whether every row has a unique, well-formed identity. A map whose
   *  identity is missing or duplicated still renders; it just cannot produce
   *  or verify an exact v1 finding. */
  identity: { ok: true } | { ok: false; reason: string };
}

export interface Columns {
  meta: NebulaiMeta;
  schema: 1 | 2;
  count: number;
  pos2: Float32Array; // n*2
  pos3: Float32Array; // n*3
  clusterId: Int32Array; // n, -1 = noise
  confidence: Uint8Array; // n, quantized 0–255 (opacity only needs 8 bits)
  labels: string[]; // stays on the main thread, never uploaded
  clusters: NebulaiCluster[];
  edges: EdgeColumns | null;
  source: SourceColumns;
}

export function columnarize(doc: NebulaiDoc): Columns {
  const n = doc.points.length;
  const pos2 = new Float32Array(n * 2);
  const pos3 = new Float32Array(n * 3);
  const clusterId = new Int32Array(n);
  const confidence = new Uint8Array(n);
  const labels = new Array<string>(n);

  for (let i = 0; i < n; i++) {
    const p = doc.points[i]!;
    pos2[i * 2] = p.xy[0];
    pos2[i * 2 + 1] = p.xy[1];
    pos3[i * 3] = p.xyz[0];
    pos3[i * 3 + 1] = p.xyz[1];
    pos3[i * 3 + 2] = p.xyz[2];
    clusterId[i] = p.cluster_id;
    confidence[i] = Math.round(Math.min(Math.max(p.confidence, 0), 1) * 255);
    labels[i] = p.label;
  }

  let edges: EdgeColumns | null = null;
  if (doc.edges) {
    const ce = doc.edges.cluster_edges;
    const clusterEdges = new Float32Array(ce.length * 3);
    for (let i = 0; i < ce.length; i++) {
      clusterEdges[i * 3] = ce[i]![0];
      clusterEdges[i * 3 + 1] = ce[i]![1];
      clusterEdges[i * 3 + 2] = ce[i]![2];
    }
    edges = {
      space: doc.edges.space,
      metric: doc.edges.metric,
      sigma: doc.edges.sigma,
      clusterEdges,
      knn: doc.edges.knn
        ? {
            k: doc.edges.knn.k,
            sigma: doc.edges.knn.sigma,
            ids: Int32Array.from(doc.edges.knn.ids),
            sims: Float32Array.from(doc.edges.knn.sims),
          }
        : null,
    };
  }

  return {
    source: sourceColumns(doc),
    meta: doc.meta,
    schema: schemaVersion(doc),
    count: n,
    pos2,
    pos3,
    clusterId,
    confidence,
    labels,
    clusters: doc.clusters,
    edges,
  };
}

const isSafeIndex = (v: unknown): v is number =>
  typeof v === "number" && Number.isSafeInteger(v) && v >= 0;

function sourceColumns(doc: NebulaiDoc): SourceColumns {
  const n = doc.points.length;
  const pointId = new Float64Array(n);
  const unitKindIdx = new Int32Array(n);
  const unitIndex = new Float64Array(n);
  const layer = new Float64Array(n);
  const xy = new Float64Array(n * 2);
  const xyz = new Float64Array(n * 3);
  const membership = new Float64Array(n);
  const unitKinds: string[] = [];
  const kindIds = new Map<string, number>();
  const problems: string[] = [];
  const seenIds = new Set<number>();
  const seenUnits = new Set<string>();
  const note = (m: string) => {
    if (problems.length < 3) problems.push(m);
  };
  let nProblems = 0;

  for (let i = 0; i < n; i++) {
    const p = doc.points[i]!;
    const id = (p as { id?: unknown }).id;
    if (isSafeIndex(id)) {
      pointId[i] = id;
      if (seenIds.has(id)) {
        nProblems++;
        note(`duplicate point id ${id}`);
      }
      seenIds.add(id);
    } else {
      pointId[i] = NaN;
      nProblems++;
      note(`row ${i} has no valid point id`);
    }
    const ref = (p as { unit_ref?: { kind?: unknown; index?: unknown } }).unit_ref;
    const kind = ref && typeof ref.kind === "string" && ref.kind.length > 0 ? ref.kind : null;
    const idx = ref && isSafeIndex(ref.index) ? ref.index : null;
    if (kind === null || idx === null) {
      unitKindIdx[i] = -1;
      unitIndex[i] = NaN;
      nProblems++;
      note(`row ${i} has no valid unit_ref`);
    } else {
      let k = kindIds.get(kind);
      if (k === undefined) {
        k = unitKinds.length;
        unitKinds.push(kind);
        kindIds.set(kind, k);
      }
      unitKindIdx[i] = k;
      unitIndex[i] = idx;
      const key = `${k}:${idx}`;
      if (seenUnits.has(key)) {
        nProblems++;
        note(`duplicate unit ${kind}#${idx}`);
      }
      seenUnits.add(key);
    }
    layer[i] = typeof p.layer === "number" && Number.isFinite(p.layer) ? p.layer : NaN;
    xy[i * 2] = p.xy[0];
    xy[i * 2 + 1] = p.xy[1];
    xyz[i * 3] = p.xyz[0];
    xyz[i * 3 + 1] = p.xyz[1];
    xyz[i * 3 + 2] = p.xyz[2];
    membership[i] = p.confidence;
  }

  return {
    pointId,
    unitKindIdx,
    unitKinds,
    unitIndex,
    layer,
    xy,
    xyz,
    membership,
    identity:
      nProblems === 0
        ? { ok: true }
        : {
            ok: false,
            reason: `${problems.join("; ")}${nProblems > problems.length ? ` (+${nProblems - problems.length} more)` : ""}`,
          },
  };
}

/** One row's immutable source identity and evidence, as exported. */
export interface SourceUnit {
  row: number;
  pointId: number | null;
  unitKind: string | null;
  unitIndex: number | null;
  layer: number | null;
  label: string;
  clusterId: number;
  membership: number;
  xy: [number, number];
  xyz: [number, number, number];
}

const fin = (v: number): number | null => (Number.isNaN(v) ? null : v);

export function sourceUnit(c: Columns, row: number): SourceUnit | null {
  if (!Number.isInteger(row) || row < 0 || row >= c.count) return null;
  const s = c.source;
  const k = s.unitKindIdx[row]!;
  return {
    row,
    pointId: fin(s.pointId[row]!),
    unitKind: k >= 0 ? s.unitKinds[k]! : null,
    unitIndex: fin(s.unitIndex[row]!),
    layer: fin(s.layer[row]!),
    label: c.labels[row] ?? "",
    clusterId: c.clusterId[row]!,
    membership: s.membership[row]!,
    xy: [s.xy[row * 2]!, s.xy[row * 2 + 1]!],
    xyz: [s.xyz[row * 3]!, s.xyz[row * 3 + 1]!, s.xyz[row * 3 + 2]!],
  };
}

/** Resolve an exact (point id, unit kind, unit index) tuple to its row.
 *  Labels and row order are never identities. */
export function findSourceRow(
  c: Columns,
  pointId: number,
  unitKind: string,
  unitIndex: number,
): { row: number } | { error: "missing" | "ambiguous" } {
  const s = c.source;
  const k = s.unitKinds.indexOf(unitKind);
  if (k < 0) return { error: "missing" };
  let found = -1;
  let hits = 0;
  for (let i = 0; i < c.count; i++) {
    if (s.pointId[i] === pointId && s.unitKindIdx[i] === k && s.unitIndex[i] === unitIndex) {
      hits++;
      found = i;
    }
  }
  if (hits === 0) return { error: "missing" };
  if (hits > 1) return { error: "ambiguous" };
  return { row: found };
}

/** The ArrayBuffers to hand to postMessage as transferables (zero-copy). */
export function transferables(c: Columns): ArrayBuffer[] {
  const bufs = [
    c.pos2.buffer,
    c.pos3.buffer,
    c.clusterId.buffer,
    c.confidence.buffer,
    c.source.pointId.buffer,
    c.source.unitKindIdx.buffer,
    c.source.unitIndex.buffer,
    c.source.layer.buffer,
    c.source.xy.buffer,
    c.source.xyz.buffer,
    c.source.membership.buffer,
  ];
  if (c.edges) {
    bufs.push(c.edges.clusterEdges.buffer);
    if (c.edges.knn) bufs.push(c.edges.knn.ids.buffer, c.edges.knn.sims.buffer);
  }
  return bufs as ArrayBuffer[];
}
