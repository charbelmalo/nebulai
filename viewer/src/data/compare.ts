/** Cross-model comparison data (`out/compare/compare.json`). Each point is a
 *  cluster centroid from one model, carrying its position in every layout
 *  state — the CompareDriver interpolates between states on the GPU. At ~840
 *  points the file is small (~400KB): plain fetch, no worker, and the CPU
 *  hover loop the atlas outgrew is perfectly fine here. */

export interface ComparePoint {
  source: string;
  source_idx: number;
  title: string;
  size: number;
  meta_cluster: number;
  shared: boolean;
  color: [number, number, number];
  positions: Record<string, [number, number, number]>;
}

export interface CompareData {
  meta: {
    models: string[];
    n_points: number;
    n_meta_clusters: number;
    embed_model: string;
  };
  states: string[];
  colors: Record<string, [number, number, number]>;
  stats: {
    n_shared_concepts: number;
    n_unique_per_model: Record<string, number>;
    jaccard: Record<string, number>;
  };
  points: ComparePoint[];
}

/** floats per instance: 4 states × vec3 + color vec3 + size + srcIdx + shared */
export const COMPARE_FLOATS = 18;

/** Instance buffer layout mirrors backend/viewer.py so the WGSL port stays a
 *  line-for-line translation. Size is sqrt-scaled against the largest cluster. */
export function compareInstances(data: CompareData): Float32Array {
  const pts = data.points;
  const states = data.states;
  let maxSize = 1;
  for (const p of pts) maxSize = Math.max(maxSize, p.size);

  const inst = new Float32Array(pts.length * COMPARE_FLOATS);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    const o = i * COMPARE_FLOATS;
    for (let s = 0; s < 4; s++) {
      const pos = p.positions[states[s]!]!;
      inst[o + s * 3] = pos[0];
      inst[o + s * 3 + 1] = pos[1];
      inst[o + s * 3 + 2] = pos[2];
    }
    inst[o + 12] = p.color[0];
    inst[o + 13] = p.color[1];
    inst[o + 14] = p.color[2];
    inst[o + 15] = 0.28 + 0.95 * Math.sqrt(p.size / maxSize);
    inst[o + 16] = p.source_idx;
    inst[o + 17] = p.shared ? 1 : 0;
  }
  return inst;
}

let cached: CompareData | null | undefined;

/** null = no comparison exported yet (run `nebulai compare` first). */
export async function loadCompare(base = "/out"): Promise<CompareData | null> {
  if (cached !== undefined) return cached;
  try {
    const res = await fetch(`${base}/compare/compare.json`);
    cached = res.ok ? ((await res.json()) as CompareData) : null;
  } catch {
    cached = null;
  }
  return cached;
}
